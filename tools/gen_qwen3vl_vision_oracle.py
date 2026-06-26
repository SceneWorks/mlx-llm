#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

SNAPSHOT = Path(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
).expanduser()
TAPS = [8, 16, 24]
GRID_THW = [1, 4, 4]


def vision_cu_seqlens(grid_thw):
    cu = [0]
    acc = 0
    for t, h, w in grid_thw:
        frame = h * w
        for _ in range(t):
            acc += frame
            cu.append(acc)
    return cu


def vision_position_ids(grid_thw, merge):
    out = []
    for t, h, w in grid_thw:
        hm, wm = h // merge, w // merge
        per = []
        for a in range(hm):
            for c in range(wm):
                for b in range(merge):
                    for d in range(merge):
                        per.append((a * merge + b, c * merge + d))
        for _ in range(t):
            out.extend(per)
    return [x for rc in out for x in rc]


def vision_bilinear(grid_thw, side, merge):
    idx = [[], [], [], []]
    wts = [[], [], [], []]
    for t, h, w in grid_thw:
        def lin(length):
            if length == 1:
                return [0.0]
            return [(side - 1) * i / (length - 1) for i in range(length)]

        h_grid = lin(h)
        w_grid = lin(w)
        h_floor = [int(x) for x in h_grid]
        w_floor = [int(x) for x in w_grid]
        h_ceil = [min(x + 1, side - 1) for x in h_floor]
        w_ceil = [min(x + 1, side - 1) for x in w_floor]
        h_frac = [g - f for g, f in zip(h_grid, h_floor)]
        w_frac = [g - f for g, f in zip(w_grid, w_floor)]

        hw = h * w
        c_idx = [[0] * hw for _ in range(4)]
        c_w = [[0.0] * hw for _ in range(4)]
        for i in range(h):
            for j in range(w):
                p = i * w + j
                hf = h_floor[i] * side
                hc = h_ceil[i] * side
                c_idx[0][p] = hf + w_floor[j]
                c_idx[1][p] = hf + w_ceil[j]
                c_idx[2][p] = hc + w_floor[j]
                c_idx[3][p] = hc + w_ceil[j]
                hfr, wfr = h_frac[i], w_frac[j]
                c_w[0][p] = (1.0 - hfr) * (1.0 - wfr)
                c_w[1][p] = (1.0 - hfr) * wfr
                c_w[2][p] = hfr * (1.0 - wfr)
                c_w[3][p] = hfr * wfr

        hm, wm = h // merge, w // merge
        reorder = []
        for a in range(hm):
            for c in range(wm):
                for b in range(merge):
                    for d in range(merge):
                        reorder.append((a * merge + b) * w + (c * merge + d))
        for _ in range(t):
            for r in reorder:
                for corner in range(4):
                    idx[corner].append(c_idx[corner][r])
                    wts[corner].append(c_w[corner][r])
    return [x for corner in idx for x in corner], [x for corner in wts for x in corner]


def fixed_pixel_values(n, patch_in):
    return [((i * 37 + 17) % 1024) / 511.5 - 1.0 for i in range(n * patch_in)]


def load_vision_model(snapshot):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, Qwen3VLVisionModel

    config = AutoConfig.from_pretrained(snapshot, local_files_only=True).vision_config
    config._attn_implementation = "eager"
    model = Qwen3VLVisionModel(config)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    shards = sorted(
        {v for k, v in index["weight_map"].items() if k.startswith("model.visual.")}
    )
    state = {}
    for shard in shards:
        with safe_open(snapshot / shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("model.visual."):
                    state[key.removeprefix("model.visual.")] = f.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    model.eval()
    model.to(dtype=torch.float32)
    return config, model


def tensor_list(tensor):
    return [round(float(x), 8) for x in tensor.detach().cpu().float().reshape(-1).tolist()]


def oracle(snapshot):
    import torch

    config, model = load_vision_model(snapshot)
    dims = {
        "depth": config.depth,
        "hidden": config.hidden_size,
        "num_heads": config.num_heads,
        "head_dim": config.hidden_size // config.num_heads,
        "inter": config.intermediate_size,
        "in_ch": config.in_channels,
        "patch": config.patch_size,
        "tpatch": config.temporal_patch_size,
        "merge": config.spatial_merge_size,
        "out_hidden": config.out_hidden_size,
        "num_pos": config.num_position_embeddings,
        "grid_per_side": int(config.num_position_embeddings ** 0.5),
        "patch_in": config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size,
        "N": GRID_THW[0] * GRID_THW[1] * GRID_THW[2],
    }
    pixel = fixed_pixel_values(dims["N"], dims["patch_in"])
    pixel_values = torch.tensor(pixel, dtype=torch.float32).reshape(dims["N"], dims["patch_in"])
    grid = torch.tensor([GRID_THW], dtype=torch.long)
    with torch.no_grad():
        out = model(pixel_values, grid)
    bi, bw = vision_bilinear([GRID_THW], dims["grid_per_side"], dims["merge"])
    return {
        "source": "tools/gen_qwen3vl_vision_oracle.py",
        "mode": "hf_qwen3vl_vision_numeric_oracle",
        "hf_model": "Qwen/Qwen3-VL-8B-Instruct",
        "hf_revision": snapshot.name,
        "dtype": "float32_reference_from_bfloat16_weights",
        "deepstack_visual_indexes": list(config.deepstack_visual_indexes),
        "dims": dims,
        "grid_thw": GRID_THW,
        "pixel": [round(x, 8) for x in pixel],
        "expected_output_shape": list(out.pooler_output.shape),
        "expected_output": tensor_list(out.pooler_output),
        "expected_deepstack_shapes": [list(x.shape) for x in out.deepstack_features],
        "expected_deepstack": [tensor_list(x) for x in out.deepstack_features],
        "expect_cu": vision_cu_seqlens([GRID_THW]),
        "expect_pos_ids": vision_position_ids([GRID_THW], dims["merge"]),
        "expect_bi": bi,
        "expect_bw": bw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=os.environ.get("QWEN3VL_SNAPSHOT", str(SNAPSHOT)))
    parser.add_argument("--out", default="src/models/testdata/qwen3vl_vision_oracle.json")
    args = parser.parse_args()
    snapshot = Path(args.snapshot).expanduser()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(oracle(snapshot), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
