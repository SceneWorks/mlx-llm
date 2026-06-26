use mlx_rs::ops::{add, concatenate_axis};
use mlx_rs::Array;

use crate::error::{Error, Result};

/// The interleaved M-RoPE position rows plus the `mrope_delta` continuation offset. See
/// [`mrope_positions_mm`].
pub type MropePositions = (Vec<i32>, Vec<i32>, Vec<i32>, i32);

/// Replace the `image_token_id` rows of `embeds` `[1, S, hidden]` with `image_features`
/// `[num_image_tokens, hidden]` (the vision encoder's merged patch rows), in sequence order — the
/// shared VLM splice (no scatter; contiguous text/image spans concatenated). `hidden` is the decoder
/// width. The number of image-token positions must equal the feature-row count.
pub(crate) fn splice_image_features(
    embeds: &Array,
    input_ids: &[i32],
    image_features: &Array,
    image_token_id: i32,
    hidden: i32,
    compute_dtype: mlx_rs::Dtype,
) -> Result<Array> {
    splice_vision_features(embeds, input_ids, image_features, &[image_token_id], hidden, compute_dtype)
}

/// Generalised VLM splice: replace every row whose id is **any** of `placeholder_tokens` (image
/// `<|image_pad|>` and/or video `<|video_pad|>`) with the next feature row, in sequence order. This is
/// the multimodal splice for a mixed image+video prompt — the visual features (image features then
/// the video's per-frame merged rows, concatenated in the same order the placeholders appear) line up
/// one-to-one with the visual positions. Reduces to [`splice_image_features`] for a single token.
pub(crate) fn splice_vision_features(
    embeds: &Array,
    input_ids: &[i32],
    vision_features: &Array,
    placeholder_tokens: &[i32],
    hidden: i32,
    compute_dtype: mlx_rs::Dtype,
) -> Result<Array> {
    let s = embeds.shape()[1] as usize;
    let feats = vision_features.as_dtype(compute_dtype)?;
    let is_vis = |id: i32| placeholder_tokens.contains(&id);
    let num_vis = input_ids.iter().filter(|&&x| is_vis(x)).count() as i32;
    if num_vis != feats.shape()[0] {
        return Err(Error::Msg(format!(
            "vlm splice: {num_vis} vision tokens != {} feature rows",
            feats.shape()[0]
        )));
    }
    if num_vis == 0 {
        return Ok(embeds.clone());
    }
    let mut pieces: Vec<Array> = Vec::new();
    let mut feat_off = 0i32;
    let mut i = 0usize;
    while i < s {
        let vis = is_vis(input_ids[i]);
        let mut j = i;
        while j < s && is_vis(input_ids[j]) == vis {
            j += 1;
        }
        let n = (j - i) as i32;
        if vis {
            let idx = Array::from_slice(&(feat_off..feat_off + n).collect::<Vec<_>>(), &[n]);
            pieces.push(feats.take_axis(&idx, 0)?.reshape(&[1, n, hidden])?);
            feat_off += n;
        } else {
            let idx = Array::from_slice(&(i as i32..j as i32).collect::<Vec<_>>(), &[n]);
            pieces.push(embeds.take_axis(&idx, 1)?);
        }
        i = j;
    }
    let refs: Vec<&Array> = pieces.iter().collect();
    Ok(concatenate_axis(&refs, 1)?)
}

/// The Qwen3-VL `get_rope_index` port (B=1) over image **and** video vision runs, returning the
/// interleaved-M-RoPE `(t, h, w)` rows (each length `S`) plus the `mrope_delta`
/// (`max_position + 1 − len`) the decode loop adds to continue positions after the prompt.
///
/// Text tokens advance all three axes by 1. A vision run lays its tokens over the
/// `(t, h/merge, w/merge)` grid offset by the shared cursor `cur` (`= max(prev) + 1`), then advances
/// the cursor by `max(grid_t, h/merge, w/merge)`. Qwen3-VL emits one video-token run **per frame**
/// (timestamp-separated), so each `[t, h, w]` video grid expands to `t × [1, h, w]` per-frame blocks.
/// Image grids are consumed one run per `image_grid_thw` entry (always `gt = 1`).
pub(crate) fn mrope_positions_mm(
    input_ids: &[i32],
    image_grid_thw: &[[i32; 3]],
    image_token_id: i32,
    video_grid_thw: &[[i32; 3]],
    video_token_id: i32,
    spatial_merge_size: i32,
) -> Result<MropePositions> {
    let merge = spatial_merge_size.max(1);
    let mut video_frames: Vec<[i32; 3]> = Vec::new();
    for &[t, h, w] in video_grid_thw {
        if t <= 0 {
            return Err(Error::Msg(format!("vlm mrope: bad video grid {:?}", [t, h, w])));
        }
        for _ in 0..t {
            video_frames.push([1, h, w]);
        }
    }

    let (mut t, mut h, mut w) = (Vec::new(), Vec::new(), Vec::new());
    let mut cur = 0i32;
    let (mut img_i, mut vid_i) = (0usize, 0usize);
    let mut i = 0usize;
    while i < input_ids.len() {
        let id = input_ids[i];
        let is_image = id == image_token_id;
        let is_video = !is_image && id == video_token_id;
        if is_image || is_video {
            let (grid, label): ([i32; 3], &str) = if is_image {
                let g = *image_grid_thw.get(img_i).ok_or_else(|| {
                    Error::Msg("vlm mrope: more image runs than image_grid_thw entries".into())
                })?;
                img_i += 1;
                (g, "image")
            } else {
                let g = *video_frames.get(vid_i).ok_or_else(|| {
                    Error::Msg("vlm mrope: more video frame runs than video_grid_thw frames".into())
                })?;
                vid_i += 1;
                (g, "video")
            };
            let (gt, gh, gw) = (grid[0], grid[1] / merge, grid[2] / merge);
            if gt <= 0 || gh <= 0 || gw <= 0 {
                return Err(Error::Msg(format!("vlm mrope: bad {label} grid {grid:?}")));
            }
            let count = (gt * gh * gw) as usize;
            let run = input_ids[i..].iter().take_while(|&&x| x == id).count();
            if run != count {
                return Err(Error::Msg(format!(
                    "vlm mrope: {label} run length {run} != grid tokens {count}"
                )));
            }
            let frame = gh * gw;
            for k in 0..count as i32 {
                t.push(k / frame + cur);
                let rem = k % frame;
                h.push(rem / gw + cur);
                w.push(rem % gw + cur);
            }
            cur += gt.max(gh).max(gw);
            i += count;
        } else {
            t.push(cur);
            h.push(cur);
            w.push(cur);
            cur += 1;
            i += 1;
        }
    }
    let maxpos = t.iter().chain(h.iter()).chain(w.iter()).copied().max().unwrap_or(-1);
    let delta = maxpos + 1 - input_ids.len() as i32;
    Ok((t, h, w, delta))
}

pub(crate) fn deepstack_fused_decoder_layers<F>(
    h0: &Array,
    visual_pos_mask: &[bool],
    deepstack: &[Array],
    num_layers: usize,
    mut layer_forward: F,
) -> Result<Array>
where
    F: FnMut(usize, &Array) -> Result<Array>,
{
    let mut h = h0.clone();
    for i in 0..num_layers {
        h = layer_forward(i, &h)?;
        if let Some(feature) = deepstack.get(i) {
            h = add_visual_features(&h, visual_pos_mask, feature)?;
        }
    }
    Ok(h)
}

pub(crate) fn add_visual_features(
    h: &Array,
    visual_pos_mask: &[bool],
    visual: &Array,
) -> Result<Array> {
    let sh = h.shape();
    let (b, s, hidden) = (sh[0], sh[1], sh[2]);
    if b != 1 {
        return Err(Error::Msg(format!(
            "deepstack fusion expects batch 1, got {b}"
        )));
    }
    if visual_pos_mask.len() != s as usize {
        return Err(Error::Msg(format!(
            "deepstack mask length {} != seq {s}",
            visual_pos_mask.len()
        )));
    }
    let num_visual = visual_pos_mask.iter().filter(|&&m| m).count() as i32;
    if num_visual != visual.shape()[0] {
        return Err(Error::Msg(format!(
            "deepstack: {num_visual} visual positions != {} feature rows",
            visual.shape()[0]
        )));
    }
    if visual.shape()[1] != hidden {
        return Err(Error::Msg(format!(
            "deepstack hidden {} != decoder hidden {hidden}",
            visual.shape()[1]
        )));
    }
    if num_visual == 0 {
        return Ok(h.clone());
    }
    let vis = visual.as_dtype(h.dtype())?;
    let mut pieces: Vec<Array> = Vec::new();
    let mut vis_off = 0i32;
    let mut i = 0usize;
    while i < s as usize {
        let is_vis = visual_pos_mask[i];
        let mut j = i;
        while j < s as usize && visual_pos_mask[j] == is_vis {
            j += 1;
        }
        let n = (j - i) as i32;
        let idx = Array::from_slice(&(i as i32..j as i32).collect::<Vec<_>>(), &[n]);
        let span = h.take_axis(&idx, 1)?;
        if is_vis {
            let vidx = Array::from_slice(&(vis_off..vis_off + n).collect::<Vec<_>>(), &[n]);
            let vspan = vis.take_axis(&vidx, 0)?.reshape(&[1, n, hidden])?;
            pieces.push(add(&span, &vspan)?);
            vis_off += n;
        } else {
            pieces.push(span);
        }
        i = j;
    }
    let refs: Vec<&Array> = pieces.iter().collect();
    Ok(concatenate_axis(&refs, 1)?)
}
