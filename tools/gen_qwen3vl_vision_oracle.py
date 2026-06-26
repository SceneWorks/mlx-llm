#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

DIMS = {
    "depth": 27,
    "hidden": 1152,
    "num_heads": 16,
    "head_dim": 72,
    "inter": 4304,
    "in_ch": 3,
    "patch": 16,
    "tpatch": 2,
    "merge": 2,
    "out_hidden": 4096,
    "num_pos": 2304,
    "grid_per_side": 48,
    "patch_in": 1536,
    "N": 16,
}
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


def oracle():
    grid = [GRID_THW]
    n_tokens = DIMS["N"] // (DIMS["merge"] * DIMS["merge"])
    bi, bw = vision_bilinear(grid, DIMS["grid_per_side"], DIMS["merge"])
    return {
        "source": "tools/gen_qwen3vl_vision_oracle.py",
        "mode": "confirmed_config_shape_oracle",
        "note": "Confirmed Qwen3-VL vision geometry and DeepStack tap oracle. Full HF reference tensor generation requires transformers/torch/model weights and is intentionally not embedded as multi-GB JSON.",
        "dims": DIMS,
        "deepstack_visual_indexes": TAPS,
        "grid_thw": GRID_THW,
        "expected_output_shape": [n_tokens, DIMS["out_hidden"]],
        "expected_deepstack_shape": [n_tokens, DIMS["out_hidden"]],
        "expect_cu": vision_cu_seqlens(grid),
        "expect_pos_ids": vision_position_ids(grid, DIMS["merge"]),
        "expect_bi": bi,
        "expect_bw": bw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="src/models/testdata/qwen3vl_vision_oracle.json",
        help="Path to write the Qwen3-VL oracle JSON",
    )
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(oracle(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
