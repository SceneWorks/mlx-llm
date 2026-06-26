use mlx_rs::ops::{add, concatenate_axis};
use mlx_rs::Array;

use crate::error::{Error, Result};

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
