//! GGML quant-type dequantization: raw block bytes → dense `f32`.
//!
//! Each GGML weight type packs values in fixed-size blocks (32 elements for the legacy `Q*_0/_1`
//! types, 256 — `QK_K` — for the k-quants) with per-block (and, for k-quants, per-sub-block) scales.
//! The block layouts and dequant arithmetic here mirror llama.cpp's `ggml-quants.c` and candle's
//! `k_quants.rs` (the battle-tested Rust port) bit-for-bit, so a tensor written by `llama-quantize`
//! reconstructs to the same floats the GPU would compute against.
//!
//! Supported: `F32`, `F16`, `BF16`, legacy `Q4_0`/`Q4_1`/`Q5_0`/`Q5_1`/`Q8_0`, k-quants
//! `Q2_K`/`Q3_K`/`Q4_K`/`Q5_K`/`Q6_K`, the non-linear 4-bit `IQ4_NL`/`IQ4_XS` (a fixed 16-entry
//! codebook — common in real `Q2_K`/`Q3_K_M`/`IQ4_XS` builds, which mix it in), and the sub-4-bit
//! importance-matrix grid-codebook types `IQ2_XXS`/`IQ2_XS`/`IQ2_S`/`IQ3_XXS`/`IQ3_S`/`IQ1_S`/`IQ1_M`
//! (story 7250). The IQ grid/sign tables live in [`super::iq_grids`]; the imatrix only shaped the
//! original quantization, so dequant is self-contained — no importance matrix is needed to decode.
//! Every quant type a standard Llama/Qwen GGUF is built from now converts.

use crate::error::{Error, Result};
use crate::gguf::iq_grids::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS,
};

/// `QK_K` — elements per k-quant super-block.
const QK_K: usize = 256;
/// Elements per legacy (`Q*_0`/`Q*_1`/`Q8_0`) block.
const QK: usize = 32;
/// The grid-offset bias added to each IQ1 codebook value (llama.cpp `IQ1S_DELTA`).
const IQ1S_DELTA: f32 = 0.125;

/// A recognized GGML tensor type with its block geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    /// 32-bit float (tag 0).
    F32,
    /// 16-bit float (tag 1).
    F16,
    /// 4-bit, 32/block, symmetric (tag 2).
    Q4_0,
    /// 4-bit, 32/block, asymmetric (tag 3).
    Q4_1,
    /// 5-bit, 32/block, symmetric (tag 6).
    Q5_0,
    /// 5-bit, 32/block, asymmetric (tag 7).
    Q5_1,
    /// 8-bit, 32/block, symmetric (tag 8).
    Q8_0,
    /// 2-bit k-quant (tag 10).
    Q2K,
    /// 3-bit k-quant (tag 11).
    Q3K,
    /// 4-bit k-quant (tag 12).
    Q4K,
    /// 5-bit k-quant (tag 13).
    Q5K,
    /// 6-bit k-quant (tag 14).
    Q6K,
    /// Non-linear 4-bit, 32/block, fixed codebook (tag 20).
    Iq4Nl,
    /// Non-linear 4-bit k-quant super-block, fixed codebook (tag 23).
    Iq4Xs,
    /// 2.06-bpw imatrix grid codebook, packed scale in `qs` (tag 16).
    Iq2Xxs,
    /// 2.31-bpw imatrix grid codebook, per-sub-block scales (tag 17).
    Iq2Xs,
    /// 2.56-bpw imatrix grid codebook, high-bit + sign bytes + scales (tag 22).
    Iq2S,
    /// 3.06-bpw imatrix grid codebook, packed scale+signs in `qs` (tag 18).
    Iq3Xxs,
    /// 3.44-bpw imatrix grid codebook, high-bit + sign bytes + scales (tag 21).
    Iq3S,
    /// 1.56-bpw imatrix grid codebook, signed grid + per-sub-block scale/delta (tag 19).
    Iq1S,
    /// 1.75-bpw imatrix grid codebook, f16 scale packed across `scales` nibbles (tag 29).
    Iq1M,
    /// bfloat16 (tag 30).
    BF16,
}

/// The fixed 16-entry non-linear codebook shared by `IQ4_NL`/`IQ4_XS` (llama.cpp `kvalues_iq4nl`).
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

impl GgmlType {
    /// Map a GGUF type tag to a [`GgmlType`], or error for an unsupported one.
    pub fn from_tag(tag: u32) -> Result<Self> {
        Ok(match tag {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            10 => GgmlType::Q2K,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            16 => GgmlType::Iq2Xxs,
            17 => GgmlType::Iq2Xs,
            18 => GgmlType::Iq3Xxs,
            19 => GgmlType::Iq1S,
            20 => GgmlType::Iq4Nl,
            21 => GgmlType::Iq3S,
            22 => GgmlType::Iq2S,
            23 => GgmlType::Iq4Xs,
            29 => GgmlType::Iq1M,
            30 => GgmlType::BF16,
            other => {
                return Err(Error::Unsupported(format!("GGUF type tag {other}")))
            }
        })
    }

    /// `(elements_per_block, bytes_per_block)`.
    pub fn block(self) -> (usize, usize) {
        match self {
            GgmlType::F32 => (1, 4),
            GgmlType::F16 | GgmlType::BF16 => (1, 2),
            GgmlType::Q4_0 => (QK, 2 + QK / 2),         // d + 16
            GgmlType::Q4_1 => (QK, 2 + 2 + QK / 2),     // d + m + 16
            GgmlType::Q5_0 => (QK, 2 + 4 + QK / 2),     // d + qh + 16
            GgmlType::Q5_1 => (QK, 2 + 2 + 4 + QK / 2), // d + m + qh + 16
            GgmlType::Q8_0 => (QK, 2 + QK),             // d + 32
            GgmlType::Q2K => (QK_K, QK_K / 16 + QK_K / 4 + 2 + 2), // scales + qs + d + dmin = 84
            GgmlType::Q3K => (QK_K, QK_K / 8 + QK_K / 4 + 12 + 2), // hmask + qs + scales + d = 110
            GgmlType::Q4K => (QK_K, 2 + 2 + 12 + QK_K / 2), // d + dmin + scales + qs = 144
            GgmlType::Q5K => (QK_K, 2 + 2 + 12 + QK_K / 8 + QK_K / 2), // + qh = 176
            GgmlType::Q6K => (QK_K, QK_K / 2 + QK_K / 4 + QK_K / 16 + 2), // ql + qh + scales + d = 210
            GgmlType::Iq4Nl => (QK, 2 + QK / 2),                 // d + qs = 18
            GgmlType::Iq4Xs => (QK_K, 2 + 2 + QK_K / 64 + QK_K / 2), // d + scales_h + scales_l + qs = 136
            GgmlType::Iq2Xxs => (QK_K, 2 + QK_K / 8 * 2),        // d + qs(u16) = 66
            GgmlType::Iq2Xs => (QK_K, 2 + QK_K / 8 * 2 + QK_K / 32), // d + qs(u16) + scales = 74
            GgmlType::Iq2S => (QK_K, 2 + QK_K / 4 + QK_K / 32 + QK_K / 32), // d + qs + qh + scales = 82
            GgmlType::Iq3Xxs => (QK_K, 2 + 3 * (QK_K / 8)),      // d + qs(grid+signs) = 98
            GgmlType::Iq3S => (QK_K, 2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + QK_K / 64), // d+qs+qh+signs+scales = 110
            GgmlType::Iq1S => (QK_K, 2 + QK_K / 8 + QK_K / 32 * 2), // d + qs + qh(u16) = 50
            GgmlType::Iq1M => (QK_K, QK_K / 8 + QK_K / 16 + QK_K / 32), // qs + qh + scales = 56 (scale packed in scales)
        }
    }
}

/// Byte length of a tensor's data for the given GGML type tag and element count.
pub fn tensor_byte_len(tag: u32, num_elements: usize) -> Result<usize> {
    let ty = GgmlType::from_tag(tag)?;
    let (be, bb) = ty.block();
    if !num_elements.is_multiple_of(be) {
        return Err(Error::Msg(format!(
            "gguf: {num_elements} elements not a multiple of {be} for {ty:?}"
        )));
    }
    Ok((num_elements / be) * bb)
}

/// Dequantize a tensor's raw block bytes to dense `f32` of length `num_elements`.
pub fn dequantize(tag: u32, data: &[u8], num_elements: usize) -> Result<Vec<f32>> {
    let ty = GgmlType::from_tag(tag)?;
    let (be, bb) = ty.block();
    if !num_elements.is_multiple_of(be) {
        return Err(Error::Msg(format!(
            "gguf: {num_elements} elements not a multiple of {be} for {ty:?}"
        )));
    }
    let nblocks = num_elements / be;
    let need = nblocks * bb;
    if data.len() < need {
        return Err(Error::Msg(format!(
            "gguf: {ty:?} needs {need} bytes for {num_elements} elements, got {}",
            data.len()
        )));
    }
    let mut out = vec![0f32; num_elements];
    match ty {
        GgmlType::F32 => {
            for (o, c) in out.iter_mut().zip(data.chunks_exact(4)) {
                *o = f32::from_le_bytes(c.try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for (o, c) in out.iter_mut().zip(data.chunks_exact(2)) {
                *o = half::f16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32();
            }
        }
        GgmlType::BF16 => {
            for (o, c) in out.iter_mut().zip(data.chunks_exact(2)) {
                *o = half::bf16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32();
            }
        }
        GgmlType::Q4_0 => blocks(data, &mut out, be, bb, dequant_q4_0),
        GgmlType::Q4_1 => blocks(data, &mut out, be, bb, dequant_q4_1),
        GgmlType::Q5_0 => blocks(data, &mut out, be, bb, dequant_q5_0),
        GgmlType::Q5_1 => blocks(data, &mut out, be, bb, dequant_q5_1),
        GgmlType::Q8_0 => blocks(data, &mut out, be, bb, dequant_q8_0),
        GgmlType::Q2K => blocks(data, &mut out, be, bb, dequant_q2_k),
        GgmlType::Q3K => blocks(data, &mut out, be, bb, dequant_q3_k),
        GgmlType::Q4K => blocks(data, &mut out, be, bb, dequant_q4_k),
        GgmlType::Q5K => blocks(data, &mut out, be, bb, dequant_q5_k),
        GgmlType::Q6K => blocks(data, &mut out, be, bb, dequant_q6_k),
        GgmlType::Iq4Nl => blocks(data, &mut out, be, bb, dequant_iq4_nl),
        GgmlType::Iq4Xs => blocks(data, &mut out, be, bb, dequant_iq4_xs),
        GgmlType::Iq2Xxs => blocks(data, &mut out, be, bb, dequant_iq2_xxs),
        GgmlType::Iq2Xs => blocks(data, &mut out, be, bb, dequant_iq2_xs),
        GgmlType::Iq2S => blocks(data, &mut out, be, bb, dequant_iq2_s),
        GgmlType::Iq3Xxs => blocks(data, &mut out, be, bb, dequant_iq3_xxs),
        GgmlType::Iq3S => blocks(data, &mut out, be, bb, dequant_iq3_s),
        GgmlType::Iq1S => blocks(data, &mut out, be, bb, dequant_iq1_s),
        GgmlType::Iq1M => blocks(data, &mut out, be, bb, dequant_iq1_m),
    }
    Ok(out)
}

/// Apply a per-block dequant closure across the tensor: each call gets one block's bytes and the
/// matching `block_elems`-wide output window.
fn blocks(data: &[u8], out: &mut [f32], be: usize, bb: usize, mut f: impl FnMut(&[u8], &mut [f32])) {
    for (b, ys) in out.chunks_mut(be).enumerate() {
        f(&data[b * bb..b * bb + bb], ys);
    }
}

/// Read a little-endian f16 at byte offset `i`.
#[inline]
fn rd_f16(b: &[u8], i: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([b[i], b[i + 1]])).to_f32()
}

// ----- legacy 32-element blocks -----

fn dequant_q4_0(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qs = &b[2..2 + 16];
    for j in 0..16 {
        let x0 = (qs[j] & 0x0F) as i32 - 8;
        let x1 = (qs[j] >> 4) as i32 - 8;
        y[j] = x0 as f32 * d;
        y[j + 16] = x1 as f32 * d;
    }
}

fn dequant_q4_1(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let m = rd_f16(b, 2);
    let qs = &b[4..4 + 16];
    for j in 0..16 {
        y[j] = (qs[j] & 0x0F) as f32 * d + m;
        y[j + 16] = (qs[j] >> 4) as f32 * d + m;
    }
}

fn dequant_q5_0(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    let qs = &b[6..6 + 16];
    for j in 0..16 {
        let xh0 = (((qh >> j) << 4) & 0x10) as u8;
        let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
        let x0 = ((qs[j] & 0x0F) | xh0) as i32 - 16;
        let x1 = ((qs[j] >> 4) | xh1) as i32 - 16;
        y[j] = x0 as f32 * d;
        y[j + 16] = x1 as f32 * d;
    }
}

fn dequant_q5_1(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let m = rd_f16(b, 2);
    let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let qs = &b[8..8 + 16];
    for j in 0..16 {
        let xh0 = (((qh >> j) << 4) & 0x10) as u8;
        let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
        let x0 = ((qs[j] & 0x0F) | xh0) as f32;
        let x1 = ((qs[j] >> 4) | xh1) as f32;
        y[j] = x0 * d + m;
        y[j + 16] = x1 * d + m;
    }
}

fn dequant_q8_0(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qs = &b[2..2 + 32];
    for j in 0..32 {
        y[j] = (qs[j] as i8) as f32 * d;
    }
}

// ----- k-quants (QK_K = 256) -----

fn dequant_q2_k(b: &[u8], y: &mut [f32]) {
    let scales = &b[0..16];
    let qs = &b[16..16 + 64];
    let d = rd_f16(b, 80);
    let dmin = rd_f16(b, 82);
    let mut is = 0usize;
    let mut yi = 0usize;
    for n in (0..QK_K).step_by(128) {
        let q = &qs[n / 4..];
        let mut shift = 0u32;
        for _ in 0..4 {
            let sc = scales[is];
            is += 1;
            let dl = d * (sc & 0xF) as f32;
            let ml = dmin * (sc >> 4) as f32;
            for &qb in &q[0..16] {
                y[yi] = dl * ((qb >> shift) & 3) as f32 - ml;
                yi += 1;
            }
            let sc = scales[is];
            is += 1;
            let dl = d * (sc & 0xF) as f32;
            let ml = dmin * (sc >> 4) as f32;
            for &qb in &q[16..32] {
                y[yi] = dl * ((qb >> shift) & 3) as f32 - ml;
                yi += 1;
            }
            shift += 2;
        }
    }
}

fn dequant_q3_k(b: &[u8], y: &mut [f32]) {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let hmask = &b[0..32];
    let qs = &b[32..32 + 64];
    let scales_raw = &b[96..96 + 12];
    let d_all = rd_f16(b, 108);

    // Reconstruct 16 signed 6-bit scales (each offset by 32) from the packed 12 bytes.
    let mut aux = [
        u32::from_le_bytes([scales_raw[0], scales_raw[1], scales_raw[2], scales_raw[3]]),
        u32::from_le_bytes([scales_raw[4], scales_raw[5], scales_raw[6], scales_raw[7]]),
        u32::from_le_bytes([scales_raw[8], scales_raw[9], scales_raw[10], scales_raw[11]]),
        0,
    ];
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let mut scales = [0i8; 16];
    for (k, s) in scales.iter_mut().enumerate() {
        *s = aux[k / 4].to_le_bytes()[k % 4] as i8;
    }

    let mut is = 0usize;
    let mut yi = 0usize;
    let mut m = 1u8;
    let mut q_off = 0usize;
    for _ in (0..QK_K).step_by(128) {
        let mut shift = 0u32;
        for _ in 0..4 {
            let dl = d_all * (scales[is] as i32 - 32) as f32;
            is += 1;
            for l in 0..16 {
                let hbit = if hmask[l] & m != 0 { 0 } else { 4 };
                y[yi] = dl * (((qs[q_off + l] >> shift) & 3) as i32 - hbit) as f32;
                yi += 1;
            }
            let dl = d_all * (scales[is] as i32 - 32) as f32;
            is += 1;
            for l in 0..16 {
                let hbit = if hmask[l + 16] & m != 0 { 0 } else { 4 };
                y[yi] = dl * (((qs[q_off + l + 16] >> shift) & 3) as i32 - hbit) as f32;
                yi += 1;
            }
            shift += 2;
            m <<= 1;
        }
        q_off += 32;
    }
}

/// Reconstruct the 6-bit scale `d` and min `m` for sub-block `j` from a k-quant `scales[12]`.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

fn dequant_q4_k(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let dmin = rd_f16(b, 2);
    let scales = &b[4..4 + 12];
    let qs = &b[16..16 + 128];
    let mut is = 0usize;
    let mut yi = 0usize;
    for j in (0..QK_K).step_by(64) {
        let q = &qs[j / 2..j / 2 + 32];
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * sc as f32;
        let m1 = dmin * m as f32;
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc as f32;
        let m2 = dmin * m as f32;
        for &qb in q {
            y[yi] = d1 * (qb & 0xF) as f32 - m1;
            yi += 1;
        }
        for &qb in q {
            y[yi] = d2 * (qb >> 4) as f32 - m2;
            yi += 1;
        }
        is += 2;
    }
}

fn dequant_q5_k(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let dmin = rd_f16(b, 2);
    let scales = &b[4..4 + 12];
    let qh = &b[16..16 + 32];
    let qs = &b[48..48 + 128];
    let mut is = 0usize;
    let mut yi = 0usize;
    let mut u1 = 1u8;
    let mut u2 = 2u8;
    for j in (0..QK_K).step_by(64) {
        let ql = &qs[j / 2..j / 2 + 32];
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * sc as f32;
        let m1 = dmin * m as f32;
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc as f32;
        let m2 = dmin * m as f32;
        for l in 0..32 {
            let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
            y[yi] = d1 * ((ql[l] & 0xF) as i32 + hi) as f32 - m1;
            yi += 1;
        }
        for l in 0..32 {
            let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
            y[yi] = d2 * ((ql[l] >> 4) as i32 + hi) as f32 - m2;
            yi += 1;
        }
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

fn dequant_q6_k(b: &[u8], y: &mut [f32]) {
    let ql = &b[0..128];
    let qh = &b[128..128 + 64];
    let scales = &b[192..192 + 16];
    let d = rd_f16(b, 208);
    for n in (0..QK_K).step_by(128) {
        let idx = n / 128;
        let qlp = &ql[64 * idx..];
        let qhp = &qh[32 * idx..];
        let sco = 8 * idx;
        for l in 0..32 {
            let is = l / 16;
            let q1 = (((qlp[l] & 0x0F) as i8) | (((qhp[l] & 3) << 4) as i8)) - 32;
            let q2 = (((qlp[l + 32] & 0x0F) as i8) | ((((qhp[l] >> 2) & 3) << 4) as i8)) - 32;
            let q3 = (((qlp[l] >> 4) as i8) | ((((qhp[l] >> 4) & 3) << 4) as i8)) - 32;
            let q4 = (((qlp[l + 32] >> 4) as i8) | ((((qhp[l] >> 6) & 3) << 4) as i8)) - 32;
            y[n + l] = d * scales[sco + is] as i8 as f32 * q1 as f32;
            y[n + l + 32] = d * scales[sco + is + 2] as i8 as f32 * q2 as f32;
            y[n + l + 64] = d * scales[sco + is + 4] as i8 as f32 * q3 as f32;
            y[n + l + 96] = d * scales[sco + is + 6] as i8 as f32 * q4 as f32;
        }
    }
}

// ----- non-linear 4-bit (fixed codebook) -----

fn dequant_iq4_nl(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qs = &b[2..2 + 16];
    for j in 0..16 {
        y[j] = d * KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as f32;
        y[j + 16] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
    }
}

fn dequant_iq4_xs(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let sh = u16::from_le_bytes([b[2], b[3]]); // scales_h: 2 high bits per sub-block scale
    let scales_l = &b[4..8]; // 4 low bits per sub-block scale, two per byte
    let qs = &b[8..8 + 128];
    for ib in 0..8 {
        // 6-bit scale = 4 low bits (scales_l) | 2 high bits (scales_h), offset by 32.
        let ls = (((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32)
            | ((((sh >> (2 * ib)) & 3) as i32) << 4);
        let dl = d * (ls - 32) as f32;
        let q = &qs[ib * 16..ib * 16 + 16];
        for j in 0..16 {
            y[ib * 32 + j] = dl * KVALUES_IQ4NL[(q[j] & 0xF) as usize] as f32;
            y[ib * 32 + j + 16] = dl * KVALUES_IQ4NL[(q[j] >> 4) as usize] as f32;
        }
    }
}

// ----- sub-4-bit importance-matrix grid-codebook quants (IQ1/IQ2/IQ3) -----
//
// These mirror llama.cpp's `dequantize_row_iq*` (`ggml-quants.c`). Each 256-element super-block is
// decoded as 8 sub-blocks of 32; a grid index selects a codebook point (8 packed bytes for the
// `u64` grids, 4 for the `u32` grids) and a sign byte from `KSIGNS_IQ2XS` flips per-element signs
// via `KMASK_IQ2XS`. Grid bytes are unsigned magnitudes for IQ2/IQ3 and signed values for IQ1
// (exactly the `const uint8_t *` vs `const int8_t *` cast llama.cpp uses). The imatrix that guided
// the original quantization is not needed here — decode is fully determined by the stored bits.

/// Sign multiplier for output lane `j` given a `KSIGNS_IQ2XS` byte: `-1` where the masked bit is set.
#[inline]
fn sign(signs: u8, j: usize) -> f32 {
    if signs & KMASK_IQ2XS[j] != 0 {
        -1.0
    } else {
        1.0
    }
}

/// IQ2_XXS (tag 16): `d` + 32×`u16` `qs`. Each 32-block reads two `u32` from `qs`: the first holds
/// four 8-bit grid indices, the second packs the 4-bit block scale (top nibble) and four 7-bit sign
/// indices.
fn dequant_iq2_xxs(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let mut yi = 0usize;
    for ib32 in 0..QK_K / 32 {
        let off = 2 + 8 * ib32;
        let aux0 = u32::from_le_bytes(b[off..off + 4].try_into().unwrap());
        let aux1 = u32::from_le_bytes(b[off + 4..off + 8].try_into().unwrap());
        let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
        let idx = aux0.to_le_bytes();
        for l in 0..4 {
            let grid = IQ2XXS_GRID[idx[l] as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                y[yi + j] = db * grid[j] as f32 * sign(signs, j);
            }
            yi += 8;
        }
    }
}

/// IQ2_XS (tag 17): `d` + 32×`u16` `qs` + 8 `scales`. Each `qs` entry is a 9-bit grid index plus a
/// 7-bit sign index; the per-sub-block scale comes from the `scales` nibbles.
fn dequant_iq2_xs(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let scales = &b[66..74];
    let mut yi = 0usize;
    for (ib32, &sc) in scales.iter().enumerate() {
        let db = [
            d * (0.5 + (sc & 0xf) as f32) * 0.25,
            d * (0.5 + (sc >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let qoff = 2 + 2 * (4 * ib32 + l);
            let q = u16::from_le_bytes([b[qoff], b[qoff + 1]]);
            let grid = IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
            let dl = db[l / 2];
            for j in 0..8 {
                y[yi + j] = dl * grid[j] as f32 * sign(signs, j);
            }
            yi += 8;
        }
    }
}

/// IQ2_S (tag 22): `d` + 64 `qs` (32 grid-low bytes then 32 sign bytes) + 8 `qh` (grid high bits) +
/// 8 `scales`.
fn dequant_iq2_s(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qs = &b[2..66]; // first 32 = grid-low, next 32 = signs
    let qh = &b[66..74];
    let scales = &b[74..82];
    let mut yi = 0usize;
    for ib32 in 0..QK_K / 32 {
        let sc = scales[ib32];
        let db = [
            d * (0.5 + (sc & 0xf) as f32) * 0.25,
            d * (0.5 + (sc >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let idx = qs[4 * ib32 + l] as usize | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
            let grid = IQ2S_GRID[idx].to_le_bytes();
            let signs = qs[32 + 4 * ib32 + l];
            let dl = db[l / 2];
            for j in 0..8 {
                y[yi + j] = dl * grid[j] as f32 * sign(signs, j);
            }
            yi += 8;
        }
    }
}

/// IQ3_XXS (tag 18): `d` + 96 `qs` (64 grid-index bytes then 32 scale+sign bytes). Each 32-block
/// reads a `u32` of scale (top nibble) and four 7-bit sign indices, and eight 8-bit grid indices
/// (each grid point is 4 bytes).
fn dequant_iq3_xxs(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let ss = &b[66..98]; // scales_and_signs: one u32 per 32-block
    let mut yi = 0usize;
    for ib32 in 0..QK_K / 32 {
        let aux = u32::from_le_bytes(ss[4 * ib32..4 * ib32 + 4].try_into().unwrap());
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        let qbase = 2 + 8 * ib32;
        for l in 0..4 {
            let g1 = IQ3XXS_GRID[b[qbase + 2 * l] as usize].to_le_bytes();
            let g2 = IQ3XXS_GRID[b[qbase + 2 * l + 1] as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            for j in 0..4 {
                y[yi + j] = db * g1[j] as f32 * sign(signs, j);
                y[yi + j + 4] = db * g2[j] as f32 * sign(signs, j + 4);
            }
            yi += 8;
        }
    }
}

/// IQ3_S (tag 21): `d` + 64 `qs` + 8 `qh` (one grid high-bit per 32-block, two per `step`) + 32
/// `signs` + 4 `scales` (two 4-bit scales each). Grid points are 4 bytes (`u32` grid).
fn dequant_iq3_s(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qh = &b[66..74];
    let signs = &b[74..106];
    let scales = &b[106..110];
    let mut yi = 0usize;
    // Process 32-blocks in pairs (matching llama.cpp's `ib32 += 2`): the low scale nibble + qh byte
    // drive the first block, the high nibble + next qh byte the second.
    for k in 0..QK_K / 64 {
        let sc = scales[k];
        let db1 = d * (1 + 2 * (sc & 0xf) as i32) as f32;
        let db2 = d * (1 + 2 * (sc >> 4) as i32) as f32;
        for (half, (db, qhb)) in [(db1, qh[2 * k] as u32), (db2, qh[2 * k + 1] as u32)]
            .into_iter()
            .enumerate()
        {
            let qbase = 2 + 16 * k + 8 * half;
            let sbase = 8 * k + 4 * half;
            for l in 0..4 {
                let i1 = b[qbase + 2 * l] as usize | (((qhb << (8 - 2 * l)) & 256) as usize);
                let i2 = b[qbase + 2 * l + 1] as usize | (((qhb << (7 - 2 * l)) & 256) as usize);
                let g1 = IQ3S_GRID[i1].to_le_bytes();
                let g2 = IQ3S_GRID[i2].to_le_bytes();
                let s = signs[sbase + l];
                for j in 0..4 {
                    y[yi + j] = db * g1[j] as f32 * sign(s, j);
                    y[yi + j + 4] = db * g2[j] as f32 * sign(s, j + 4);
                }
                yi += 8;
            }
        }
    }
}

/// IQ1_S (tag 19): `d` + 32 `qs` + 8 `u16` `qh`. The grid is **signed**; each sub-block has a 3-bit
/// scale and a sign-selected `±IQ1S_DELTA` grid offset, both packed in `qh`.
fn dequant_iq1_s(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qs = &b[2..34];
    let qh = &b[34..50];
    let mut yi = 0usize;
    for ib in 0..QK_K / 32 {
        let h = u16::from_le_bytes([qh[2 * ib], qh[2 * ib + 1]]);
        let dl = d * (2 * ((h >> 12) & 7) + 1) as f32;
        let delta = if h & 0x8000 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
        for l in 0..4 {
            let idx = qs[4 * ib + l] as usize | ((((h >> (3 * l)) & 7) as usize) << 8);
            let grid = IQ1S_GRID[idx].to_le_bytes();
            for j in 0..8 {
                y[yi + j] = dl * ((grid[j] as i8) as f32 + delta);
            }
            yi += 8;
        }
    }
}

/// IQ1_M (tag 29): 32 `qs` + 16 `qh` + 8 `scales`. Unlike the others there is **no per-block `d`** —
/// the f16 super-block scale is reassembled from the top nibbles of the four `u16` `scales` words,
/// and each `scales` word also carries four 3-bit sub-scales. Grid is signed, shared with IQ1_S.
fn dequant_iq1_m(b: &[u8], y: &mut [f32]) {
    let qs = &b[0..32];
    let qh = &b[32..48];
    let sc = [
        u16::from_le_bytes([b[48], b[49]]),
        u16::from_le_bytes([b[50], b[51]]),
        u16::from_le_bytes([b[52], b[53]]),
        u16::from_le_bytes([b[54], b[55]]),
    ];
    let scale_bits =
        (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    let d = half::f16::from_bits(scale_bits).to_f32();
    let mut yi = 0usize;
    for ib in 0..QK_K / 32 {
        let w = sc[ib / 2];
        let shift = 6 * (ib % 2);
        let dl1 = d * (2 * ((w >> shift) & 0x7) + 1) as f32;
        let dl2 = d * (2 * ((w >> (shift + 3)) & 0x7) + 1) as f32;
        let qh0 = qh[2 * ib] as usize;
        let qh1 = qh[2 * ib + 1] as usize;
        let idx = [
            qs[4 * ib] as usize | ((qh0 << 8) & 0x700),
            qs[4 * ib + 1] as usize | ((qh0 << 4) & 0x700),
            qs[4 * ib + 2] as usize | ((qh1 << 8) & 0x700),
            qs[4 * ib + 3] as usize | ((qh1 << 4) & 0x700),
        ];
        let delta = [
            if qh0 & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
            if qh0 & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
            if qh1 & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
            if qh1 & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
        ];
        for l in 0..4 {
            let dl = if l < 2 { dl1 } else { dl2 };
            let grid = IQ1S_GRID[idx[l]].to_le_bytes();
            for j in 0..8 {
                y[yi + j] = dl * ((grid[j] as i8) as f32 + delta[l]);
            }
            yi += 8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_geometry_matches_llama_cpp() {
        assert_eq!(GgmlType::Q4_0.block(), (32, 18));
        assert_eq!(GgmlType::Q4_1.block(), (32, 20));
        assert_eq!(GgmlType::Q5_0.block(), (32, 22));
        assert_eq!(GgmlType::Q5_1.block(), (32, 24));
        assert_eq!(GgmlType::Q8_0.block(), (32, 34));
        assert_eq!(GgmlType::Q2K.block(), (256, 84));
        assert_eq!(GgmlType::Q3K.block(), (256, 110));
        assert_eq!(GgmlType::Q4K.block(), (256, 144));
        assert_eq!(GgmlType::Q5K.block(), (256, 176));
        assert_eq!(GgmlType::Q6K.block(), (256, 210));
        assert_eq!(GgmlType::Iq4Nl.block(), (32, 18));
        assert_eq!(GgmlType::Iq4Xs.block(), (256, 136));
        // IQ1/IQ2/IQ3 super-block sizes (the upstream `static_assert`ed `block_iq*` sizes).
        assert_eq!(GgmlType::Iq2Xxs.block(), (256, 66));
        assert_eq!(GgmlType::Iq2Xs.block(), (256, 74));
        assert_eq!(GgmlType::Iq2S.block(), (256, 82));
        assert_eq!(GgmlType::Iq3Xxs.block(), (256, 98));
        assert_eq!(GgmlType::Iq3S.block(), (256, 110));
        assert_eq!(GgmlType::Iq1S.block(), (256, 50));
        assert_eq!(GgmlType::Iq1M.block(), (256, 56));
    }

    #[test]
    fn supported_and_unsupported_types() {
        assert_eq!(GgmlType::from_tag(20).unwrap(), GgmlType::Iq4Nl);
        assert_eq!(GgmlType::from_tag(23).unwrap(), GgmlType::Iq4Xs);
        // The sub-4-bit importance-matrix IQ grid types now decode (story 7250).
        assert_eq!(GgmlType::from_tag(16).unwrap(), GgmlType::Iq2Xxs);
        assert_eq!(GgmlType::from_tag(17).unwrap(), GgmlType::Iq2Xs);
        assert_eq!(GgmlType::from_tag(18).unwrap(), GgmlType::Iq3Xxs);
        assert_eq!(GgmlType::from_tag(19).unwrap(), GgmlType::Iq1S);
        assert_eq!(GgmlType::from_tag(21).unwrap(), GgmlType::Iq3S);
        assert_eq!(GgmlType::from_tag(22).unwrap(), GgmlType::Iq2S);
        assert_eq!(GgmlType::from_tag(29).unwrap(), GgmlType::Iq1M);
        assert_eq!(tensor_byte_len(16, 256).unwrap(), 66);
        // Intermediate/quantization-only types stay unsupported.
        assert!(matches!(GgmlType::from_tag(9), Err(Error::Unsupported(_)))); // Q8_1 (intermediate)
        assert!(matches!(GgmlType::from_tag(15), Err(Error::Unsupported(_)))); // Q8_K (intermediate)
        assert!(matches!(tensor_byte_len(9, 256), Err(Error::Unsupported(_))));
    }

    /// IQ4_NL maps each 4-bit index through the fixed codebook and scales by the block's f16 `d`.
    #[test]
    fn iq4_nl_codebook_decode() {
        let mut b = Vec::new();
        b.extend_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes()); // d = 2.0
        // qs[0] = 0x80 => low nibble 0 (codebook[-127]) at y[0], high nibble 8 (codebook[1]) at y[16].
        b.push(0x80);
        b.extend([0x00u8; 15]); // both nibbles 0 => codebook[0] = -127
        let out = dequantize(20, &b, 32).unwrap();
        assert_eq!(out[0], 2.0 * -127.0); // low nibble of byte 0 = 0
        assert_eq!(out[16], 2.0 * 1.0); // high nibble of byte 0 = 8 -> codebook[8] = 1
        assert_eq!(out[1], 2.0 * -127.0); // byte 1 low nibble 0
    }

    #[test]
    fn f32_f16_bf16_passthrough() {
        let vals = [1.0f32, -2.5, 3.25, 0.0];
        let mut f32b = Vec::new();
        for v in vals {
            f32b.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(dequantize(0, &f32b, 4).unwrap(), vals);

        let mut f16b = Vec::new();
        for v in vals {
            f16b.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
        }
        assert_eq!(dequantize(1, &f16b, 4).unwrap(), vals); // these values are exact in f16

        let mut bf16b = Vec::new();
        for v in vals {
            bf16b.extend_from_slice(&half::bf16::from_f32(v).to_bits().to_le_bytes());
        }
        assert_eq!(dequantize(30, &bf16b, 4).unwrap(), vals); // exact in bf16 too
    }

    /// Q8_0: a hand-built block with scale d=0.5 and quants 0..32 dequantizes to `q * 0.5`.
    #[test]
    fn q8_0_block_dequant() {
        let mut b = Vec::new();
        b.extend_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes()); // d
        for q in 0i32..32 {
            b.push((q - 16) as i8 as u8); // quants -16..16
        }
        let out = dequantize(8, &b, 32).unwrap();
        for (i, v) in out.iter().enumerate() {
            assert!((v - (i as i32 - 16) as f32 * 0.5).abs() < 1e-3, "idx {i}: {v}");
        }
    }

    /// Q4_0: nibble layout (low nibbles → first half, high nibbles → second half), values centered
    /// at 8. A block with d=1.0 and every nibble = 8 dequantizes to all zeros.
    #[test]
    fn q4_0_zero_centered() {
        let mut b = Vec::new();
        b.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        b.extend([0x88u8; 16]); // both nibbles == 8 => (8-8)*d == 0
        let out = dequantize(2, &b, 32).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));

        // nibble 0xF in low position of byte 0 => (15-8)*1.0 = 7.0 at y[0].
        let mut b2 = Vec::new();
        b2.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        b2.push(0x8F); // low nibble F, high nibble 8
        b2.extend([0x88u8; 15]);
        let out2 = dequantize(2, &b2, 32).unwrap();
        assert_eq!(out2[0], 7.0);
        assert_eq!(out2[16], 0.0);
    }

    /// Q2_K: with super-block `d = 1`, every sub-block scale's low nibble = 1 (so `dl = 1`) and high
    /// nibble = 0 (so the min `ml = 0`), and every 2-bit quant = 3, each output is `1·3 − 0 = 3`.
    /// Exercises the scale/min split, the 2-bit unpack, and the shift schedule.
    #[test]
    fn q2_k_scale_and_unpack() {
        let mut b = Vec::new();
        b.extend([0x01u8; 16]); // scales: low nibble 1 (dl=d*1), high nibble 0 (ml=0)
        b.extend([0xFFu8; 64]); // qs: every 2-bit group == 3
        b.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes()); // d
        b.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes()); // dmin (unused: sc>>4==0)
        let out = dequantize(10, &b, 256).unwrap();
        assert!(out.iter().all(|&v| v == 3.0), "every q2_k output should be 1*3 - 0 = 3");
    }

    /// A k-quant round-trips structurally: a Q6_K block whose quant nibbles all encode the centered
    /// value (so each reconstructed q == 0) dequantizes to all zeros regardless of scales.
    #[test]
    fn q6_k_centered_is_zero() {
        // q == 0 requires (ql_nibble | (qh_bits << 4)) == 32. Build ql low-nibble=0, qh bits=2
        // (=> 0 | (2<<4) = 32 => q1 = 32 - 32 = 0). Simpler: set all bytes so every reconstructed
        // value is 32; use ql nibble 0 and qh pair bits = 2 everywhere.
        let mut b = vec![0u8; 210];
        // ql: low and high nibble both 0
        // qh: each 2-bit group = 2 => byte pattern 0b10_10_10_10 = 0xAA gives all four groups = 2.
        for x in b.iter_mut().skip(128).take(64) {
            *x = 0xAA;
        }
        // scales: arbitrary non-zero
        for x in b.iter_mut().skip(192).take(16) {
            *x = 5;
        }
        // d = 1.0
        let dbytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        b[208] = dbytes[0];
        b[209] = dbytes[1];
        let out = dequantize(14, &b, 256).unwrap();
        assert!(out.iter().all(|&v| v == 0.0), "all centered q6_k values are zero");
    }

    /// Push an f16 `d` as the first two bytes of an IQ block, then `n` zero payload bytes.
    fn iq_block_d1(total: usize) -> Vec<u8> {
        let mut b = half::f16::from_f32(1.0).to_bits().to_le_bytes().to_vec();
        b.resize(total, 0);
        b
    }

    /// IQ2_XXS: an all-zero-`qs` block with `d=1` selects grid point 0 (`0x0808…` → byte 8) at scale
    /// `(0.5+0)·0.25 = 0.125` with no sign flips, so every output is `0.125·8 = 1.0`. Setting block 0's
    /// scale nibble (top of the second `u32`) to 3 lifts only its 32 lanes to `(0.5+3)·0.25·8 = 7.0`,
    /// proving the per-32-block scale unpack and window offset.
    #[test]
    fn iq2_xxs_grid_scale_and_offset() {
        let b = iq_block_d1(66);
        let out = dequantize(16, &b, 256).unwrap();
        assert!(out.iter().all(|&v| v == 1.0), "grid-0 decode = 0.125*8 = 1.0");

        let mut b2 = iq_block_d1(66);
        b2[9] = 0x30; // second u32 of block 0 = 0x30000000 => scale nibble 3, sign indices 0
        let out2 = dequantize(16, &b2, 256).unwrap();
        assert!(out2[..32].iter().all(|&v| v == 7.0), "block 0 scaled to 7.0");
        assert_eq!(out2[32], 1.0, "block 1 unaffected");
    }

    /// IQ2_XS / IQ2_S share the grid-0 magnitude (byte 8) and the `(0.5+s)·0.25` scale; an all-zero
    /// block decodes to `0.125·8 = 1.0` everywhere (zero scales, zero sign bytes ⇒ no flips).
    #[test]
    fn iq2_xs_and_s_grid_zero_decode() {
        let out_xs = dequantize(17, &iq_block_d1(74), 256).unwrap();
        assert!(out_xs.iter().all(|&v| v == 1.0), "iq2_xs grid-0 = 1.0");
        let out_s = dequantize(22, &iq_block_d1(82), 256).unwrap();
        assert!(out_s.iter().all(|&v| v == 1.0), "iq2_s grid-0 = 1.0");
    }

    /// IQ3_XXS grid point 0 is `0x04040404` → byte 4; scale `(0.5+0)·0.5 = 0.25`, so `0.25·4 = 1.0`.
    /// IQ3_S grid point 0 is `0x01010101` → byte 1; scale `1+2·0 = 1`, so `1·1 = 1.0`.
    #[test]
    fn iq3_xxs_and_s_grid_zero_decode() {
        let out_xxs = dequantize(18, &iq_block_d1(98), 256).unwrap();
        assert!(out_xxs.iter().all(|&v| v == 1.0), "iq3_xxs grid-0 = 0.25*4 = 1.0");
        let out_s = dequantize(21, &iq_block_d1(110), 256).unwrap();
        assert!(out_s.iter().all(|&v| v == 1.0), "iq3_s grid-0 = 1*1 = 1.0");
    }

    /// IQ1_S uses a **signed** grid: point 0 is `0xffff…` → byte `-1`. With `d=1`, zero `qh` gives
    /// scale `2·0+1 = 1` and `+IQ1S_DELTA` offset, so each lane is `1·(-1 + 0.125) = -0.875`. Setting
    /// block 0's 3-bit scale (`qh` bits 12..15 = 1) triples it to `3·(-0.875) = -2.625` on its lanes.
    #[test]
    fn iq1_s_signed_grid_and_scale() {
        let out = dequantize(19, &iq_block_d1(50), 256).unwrap();
        assert!(out.iter().all(|&v| v == -0.875), "iq1_s grid-0 = 1*(-1+0.125) = -0.875");

        let mut b = iq_block_d1(50);
        b[35] = 0x10; // qh[0] = 0x1000 => (h>>12)&7 = 1 => scale 3
        let out2 = dequantize(19, &b, 256).unwrap();
        assert!(out2[..32].iter().all(|&v| v == -2.625), "block 0 scaled to -2.625");
        assert_eq!(out2[32], -0.875, "block 1 unaffected");
    }

    /// IQ1_M carries **no per-block `d`**: its f16 super-block scale is reassembled from the top
    /// nibbles of the four `u16` `scales` words. Encode `d = 1.0` (f16 `0x3C00`) across those nibbles
    /// (`sc2=0xC000`, `sc3=0x3000`, rest 0) with zero 3-bit sub-scales and zero `qs`/`qh`; the signed
    /// grid-0 byte `-1` plus `+IQ1S_DELTA` gives `1·(-1 + 0.125) = -0.875` on every lane.
    #[test]
    fn iq1_m_packed_f16_scale() {
        let mut b = vec![0u8; 56];
        b[52] = 0x00; // sc2 = 0xC000
        b[53] = 0xC0;
        b[54] = 0x00; // sc3 = 0x3000
        b[55] = 0x30;
        let out = dequantize(29, &b, 256).unwrap();
        assert!(out.iter().all(|&v| v == -0.875), "iq1_m reconstructed d=1 => -0.875");

        // A broken scale reconstruction (all-zero scales => d=0) would instead give all zeros — guard
        // that the nibble assembly is actually doing work.
        let zero = dequantize(29, &[0u8; 56], 256).unwrap();
        assert!(zero.iter().all(|&v| v == 0.0), "zero scales => d=0 => all zero");
    }
}
