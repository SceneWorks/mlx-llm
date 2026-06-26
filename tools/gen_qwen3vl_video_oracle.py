#!/usr/bin/env python3
"""Generate the Qwen3-VL **video** Text–Timestamp-Alignment numeric/layout oracle (sc-8081).

Pins, from the HF transformers reference (`processing_qwen3_vl.Qwen3VLProcessor`):

  * `replace_video_token`  — the exact per-frame placeholder *string* the processor emits for a
    sampled video: `<{t:.1f} seconds>` (Text–Timestamp Alignment) followed by
    `<|vision_start|><|video_pad|><|vision_end|>`, repeated `grid_t` times, with one `<|video_pad|>`
    per `frame_seqlen = (h·w)//merge²` (expanded after tokenizing). This is the layout mlx-llm's
    `video_placeholder_text` + the per-frame placeholder expansion must reproduce.
  * `_calculate_timestamps` — the merged per-frame timestamps from `frames_indices` + `fps`: padded
    to a multiple of `temporal_patch_size`, then averaged within each temporal patch. mlx-llm's
    `merged_frame_timestamps` mirrors this.
  * `expanded_token_ids` — tokenizing the placeholder string yields the exact id stream (timestamp
    tokens + per-frame vision framing + `video_token_id` runs). The committed ids are what
    `Qwen3VLProcessor` produces so the Rust expansion can be checked against them.

The string/timestamp math needs only the tokenizer + video_processor config from the snapshot (no
weights, no torchvision/PIL — we feed a synthetic `video_grid_thw` + `video_metadata` directly to the
processor's `replace_video_token`, exactly as the real pipeline does after frame sampling).

Self-contained: emits JSON to `src/models/testdata/qwen3vl_video_oracle.json`.
"""
import argparse
import json
import os
from pathlib import Path

SNAPSHOT = Path(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
).expanduser()


def build(snapshot):
    import numpy as np
    from transformers import AutoProcessor
    from transformers.video_utils import VideoMetadata

    proc = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    video_token = proc.video_token
    video_token_id = proc.tokenizer.convert_tokens_to_ids(video_token)
    vs = proc.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    ve = proc.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    merge = proc.video_processor.merge_size
    temporal = proc.video_processor.temporal_patch_size

    # A synthetic sampled video: 4 frames at fps=2, sampled at frame indices [0,1,2,3], a 4x4 patch
    # grid per frame (grid_h = grid_w = 4). After temporal merge (2 frames/patch): grid_t = 2.
    fps = 2.0
    frames_indices = [0, 1, 2, 3]
    grid_h, grid_w = 4, 4
    grid_t = len(frames_indices) // temporal  # = 2
    video_grid_thw = [[grid_t, grid_h, grid_w]]
    frame_seqlen = (grid_h * grid_w) // (merge * merge)

    metadata = VideoMetadata(
        total_num_frames=len(frames_indices),
        fps=fps,
        frames_indices=frames_indices,
    )
    video_inputs = {
        "video_grid_thw": np.array(video_grid_thw),
        "video_metadata": [metadata],
    }

    # The reference per-frame placeholder string (Text–Timestamp Alignment).
    placeholder = proc.replace_video_token(video_inputs, video_idx=0)

    # The merged timestamps the reference computes (one per emitted vision frame).
    merged_timestamps = proc._calculate_timestamps(frames_indices, fps, temporal)

    # Tokenize the placeholder string to the exact id stream (no special-token injection beyond what
    # the string contains — matches the engine, which tokenizes the substituted text).
    expanded_ids = proc.tokenizer.encode(placeholder, add_special_tokens=False)
    n_video_tokens = sum(1 for i in expanded_ids if i == video_token_id)
    expected_video_tokens = grid_t * frame_seqlen

    return {
        "source": "tools/gen_qwen3vl_video_oracle.py",
        "mode": "hf_qwen3vl_video_text_timestamp_alignment_oracle",
        "hf_model": "Qwen/Qwen3-VL-8B-Instruct",
        "hf_revision": snapshot.name,
        "video_token_id": int(video_token_id),
        "vision_start_token_id": int(vs),
        "vision_end_token_id": int(ve),
        "merge": int(merge),
        "temporal_patch_size": int(temporal),
        "fps": fps,
        "frames_indices": frames_indices,
        "video_grid_thw": video_grid_thw,
        "grid_t": int(grid_t),
        "grid_h": int(grid_h),
        "grid_w": int(grid_w),
        "frame_seqlen": int(frame_seqlen),
        "merged_timestamps": [round(float(t), 6) for t in merged_timestamps],
        "placeholder_text": placeholder,
        "expanded_ids": [int(i) for i in expanded_ids],
        "n_video_tokens": int(n_video_tokens),
        "expected_video_tokens": int(expected_video_tokens),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=os.environ.get("QWEN3VL_SNAPSHOT", str(SNAPSHOT)))
    parser.add_argument("--out", default="src/models/testdata/qwen3vl_video_oracle.json")
    args = parser.parse_args()
    snapshot = Path(args.snapshot).expanduser()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(snapshot), indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
