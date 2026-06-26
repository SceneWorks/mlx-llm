#!/usr/bin/env python3
"""Generate the Qwen3-VL Interleaved-MRoPE + token-expansion numeric oracle.

Pins, from the HF transformers reference (`modeling_qwen3_vl` / `processing_qwen3_vl`):

  * `rope_index_image`  — `Qwen3VLModel.get_rope_index` 3-D position rows + mrope_delta for a
    mixed text/image sequence (single image, `image_grid_thw`).
  * `rope_index_video`  — the same for a multi-frame **video** (`video_grid_thw`, the synthetic
    time axis): Qwen3-VL splits each frame into its own gt=1 vision block via timestamps, so the
    temporal index resets per frame — the Qwen3-VL-specific delta vs a single multi-t block.
  * `interleaved`       — `apply_interleaved_mrope` cos/sin for the image rope rows (head_dim 128,
    mrope_section [24,20,20], theta 5e6), validating the Qwen3-VL interleaving end to end.
  * `expand`            — the processor's image-placeholder token expansion: a chat prompt with a
    single `<|image_pad|>` framed by `<|vision_start|>` / `<|vision_end|>` expands to
    `grid.prod() // merge**2` placeholder copies. The committed `expanded_ids` are the exact ids
    `Qwen3VLProcessor` emits for the fixed grid.

The position-id math needs no weights (it is pure index arithmetic), so the oracle builds on a
meta/config-only model and runs without the multi-GB checkpoint. The interleaved cos/sin likewise
needs only the config. The processor path needs the tokenizer + image processor from the snapshot.

Self-contained: emits JSON to `src/models/testdata/qwen3vl_mrope_oracle.json`.
"""
import argparse
import json
import os
from pathlib import Path

SNAPSHOT = Path(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
).expanduser()


def rope_rows(model, input_ids, image_grid_thw=None, video_grid_thw=None):
    import torch

    ids = torch.tensor([input_ids], dtype=torch.long)
    ig = torch.tensor(image_grid_thw, dtype=torch.long) if image_grid_thw is not None else None
    vg = torch.tensor(video_grid_thw, dtype=torch.long) if video_grid_thw is not None else None
    pos, delta = model.get_rope_index(ids, image_grid_thw=ig, video_grid_thw=vg)
    rows = pos[:, 0, :].tolist()  # [3, S]  (t, h, w)
    return rows, int(delta[0, 0])


def interleaved_cos_sin(cfg, t_row, h_row, w_row):
    """Mirror Qwen3VLTextRotaryEmbedding: inv_freq · positions, apply_interleaved_mrope, cat,cos/sin."""
    import torch
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    text_cfg = cfg.text_config
    rope_type = (text_cfg.rope_scaling or {}).get("rope_type", "default")
    inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](text_cfg, torch.device("cpu"))
    mrope_section = (text_cfg.rope_scaling or {}).get("mrope_section", [24, 20, 20])

    position_ids = torch.tensor([t_row, h_row, w_row], dtype=torch.long)  # (3, S)
    position_ids = position_ids[:, None, :]  # (3, 1, S)
    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, 1, -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()  # (3, 1, 1, S)
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)  # (3, 1, S, dim/2)

    # apply_interleaved_mrope
    freqs_t = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    emb = torch.cat((freqs_t, freqs_t), dim=-1)  # (1, S, dim)
    cos = (emb.cos() * attention_scaling)[0]
    sin = (emb.sin() * attention_scaling)[0]
    return cos.reshape(-1).tolist(), sin.reshape(-1).tolist(), int(emb.shape[-1])


def processor_expand(snapshot, grid_hw):
    """Run the real processor on a single-image chat prompt; return raw + expanded ids and the grid."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    image_token = proc.image_token
    # Choose an image size that yields exactly grid_hw patches at this processor's settings. We let
    # the processor pick the grid from the image, then read back grid_thw, so the oracle is whatever
    # the real pipeline produces (no hand-tuning of resize math).
    h_px, w_px = grid_hw
    img = Image.new("RGB", (w_px, h_px), (123, 117, 104))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this."},
            ],
        }
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raw_text = text  # contains a single image_token placeholder
    enc = proc(text=[text], images=[img], return_tensors="pt")
    expanded_ids = enc["input_ids"][0].tolist()
    grid = enc["image_grid_thw"][0].tolist()
    image_token_id = proc.tokenizer.convert_tokens_to_ids(image_token)
    n_placeholders = sum(1 for i in expanded_ids if i == image_token_id)
    merge = proc.image_processor.merge_size
    expected_count = (grid[0] * grid[1] * grid[2]) // (merge * merge)
    return {
        "raw_text": raw_text,
        "expanded_ids": expanded_ids,
        "grid_thw": grid,
        "image_token_id": int(image_token_id),
        "vision_start_token_id": int(proc.tokenizer.convert_tokens_to_ids("<|vision_start|>")),
        "vision_end_token_id": int(proc.tokenizer.convert_tokens_to_ids("<|vision_end|>")),
        "merge": int(merge),
        "num_placeholders": int(n_placeholders),
        "expected_count": int(expected_count),
    }


def build(snapshot):
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    cfg = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    image_token_id = cfg.image_token_id
    video_token_id = cfg.video_token_id
    vs = cfg.vision_start_token_id
    ve = cfg.vision_end_token_id
    merge = cfg.vision_config.spatial_merge_size

    model = Qwen3VLModel(cfg)
    model.eval()

    # --- rope_index: mixed text + single image, 1x4x4 patch grid -> 2x2 merged tokens.
    img_grid = [1, 4, 4]
    img_ids = [100, 101, vs] + [image_token_id] * 4 + [ve, 102, 103]
    (it, ih, iw), idelta = rope_rows(model, img_ids, image_grid_thw=[img_grid])

    # --- rope_index: multi-frame video, 2x4x4 -> two gt=1 frame blocks (synthetic time axis).
    vid_grid = [2, 4, 4]
    vid_ids = (
        [200, vs] + [video_token_id] * 4 + [ve, vs] + [video_token_id] * 4 + [ve, 201]
    )
    (vt, vh, vw), vdelta = rope_rows(model, vid_ids, video_grid_thw=[vid_grid])

    # --- interleaved cos/sin over the image rope rows (head_dim 128, section [24,20,20], theta 5e6).
    cos, sin, dim = interleaved_cos_sin(cfg, it, ih, iw)

    # --- processor token-expansion (representative image prompt). 112x112 px -> processor grid.
    expand = processor_expand(snapshot, (112, 112))

    return {
        "source": "tools/gen_qwen3vl_mrope_oracle.py",
        "mode": "hf_qwen3vl_mrope_and_token_expansion_oracle",
        "hf_model": "Qwen/Qwen3-VL-8B-Instruct",
        "hf_revision": snapshot.name,
        "image_token_id": int(image_token_id),
        "video_token_id": int(video_token_id),
        "vision_start_token_id": int(vs),
        "vision_end_token_id": int(ve),
        "merge": int(merge),
        "head_dim": int(cfg.text_config.head_dim),
        "rope_theta": float(cfg.text_config.rope_theta),
        "mrope_section": list((cfg.text_config.rope_scaling or {}).get("mrope_section", [24, 20, 20])),
        "rope_index_image": {
            "input_ids": img_ids,
            "image_grid_thw": [img_grid],
            "t": it,
            "h": ih,
            "w": iw,
            "delta": idelta,
        },
        "rope_index_video": {
            "input_ids": vid_ids,
            "video_grid_thw": [vid_grid],
            "t": vt,
            "h": vh,
            "w": vw,
            "delta": vdelta,
        },
        "interleaved": {
            "dim": dim,
            "t": it,
            "h": ih,
            "w": iw,
            "cos": [round(x, 8) for x in cos],
            "sin": [round(x, 8) for x in sin],
        },
        "expand": expand,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=os.environ.get("QWEN3VL_SNAPSHOT", str(SNAPSHOT)))
    parser.add_argument("--out", default="src/models/testdata/qwen3vl_mrope_oracle.json")
    args = parser.parse_args()
    snapshot = Path(args.snapshot).expanduser()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(snapshot), indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
