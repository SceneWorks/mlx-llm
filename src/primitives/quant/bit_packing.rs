//! Lossless sub-byte bit-packing of integer codes (1/2/4-bit).
//!
//! KV-cache compression stores quantization codes far below a byte each. This packs `u8` codes in
//! `[0, 2^bits)` into a tight `u8` buffer and unpacks them back, losslessly, for `bits ∈ {1, 2, 4}`:
//!
//! ```text
//!   bits = 4 -> 2 codes / byte  (2× over u8 storage)
//!   bits = 2 -> 4 codes / byte  (4×)
//!   bits = 1 -> 8 codes / byte  (8×)
//! ```
//!
//! **Byte layout** (matches the VeloxQuant `_bit_packing` reference, whose algorithm we read but did
//! NOT execute): within each byte, code `i` of the group occupies bits `[i*bits, (i+1)*bits)` —
//! little-endian, lowest code in the least-significant bits:
//!
//! ```text
//!   packed_byte = Σ_i (code_i & mask) << (i * bits)
//!   code_i      = (packed_byte >> (i * bits)) & mask
//! ```
//!
//! This is a pure-Rust scatter/gather over MLX-resident `u8` data. The per-byte loop is the hot
//! path a fused kernel will collapse:
//! `TODO(sc-8529/Phase2): replace the host-side pack/unpack loops with a MetalKernel`
//! (speed only — this pure path defines correctness and is the losslessness oracle).

use mlx_rs::{Array, Dtype};

use crate::error::{Error, Result};

fn check_bits(bits: i32) -> Result<()> {
    if matches!(bits, 1 | 2 | 4) {
        Ok(())
    } else {
        Err(Error::Unsupported(format!(
            "bit_packing: bits must be 1, 2, or 4, got {bits}"
        )))
    }
}

/// Number of packed bytes needed to store `n` codes at `bits` width. `n` must be divisible by the
/// codes-per-byte (`8 / bits`).
pub fn packed_len(n: usize, bits: i32) -> Result<usize> {
    check_bits(bits)?;
    let per_byte = (8 / bits) as usize;
    if !n.is_multiple_of(per_byte) {
        return Err(Error::Msg(format!(
            "bit_packing: code count {n} not divisible by {per_byte} (= 8/bits)"
        )));
    }
    Ok(n * bits as usize / 8)
}

/// Pack a flat `u8` code array into a tight `u8` buffer.
///
/// `codes` is read as a flat `u8` array; values must be in `[0, 2^bits)` (out-of-range bits are
/// masked off, never silently corrupting neighbours). Length must be divisible by `8 / bits`.
/// Returns a 1-D `u8` [`Array`] of length [`packed_len`].
pub fn bit_pack(codes: &Array, bits: i32) -> Result<Array> {
    check_bits(bits)?;
    let codes = codes.as_dtype(Dtype::Uint8)?;
    let flat: Vec<u8> = codes.as_slice::<u8>().to_vec();
    let n = flat.len();
    let n_bytes = packed_len(n, bits)?;
    let per_byte = (8 / bits) as usize;
    let mask: u8 = ((1u16 << bits) - 1) as u8;

    let mut packed = vec![0u8; n_bytes];
    // TODO(sc-8529/Phase2): replace this host-side pack loop with a MetalKernel (one thread/byte).
    for (byte_idx, slot) in packed.iter_mut().enumerate() {
        let base = byte_idx * per_byte;
        let mut acc: u8 = 0;
        for i in 0..per_byte {
            let v = flat[base + i] & mask;
            acc |= v << (i as u32 * bits as u32);
        }
        *slot = acc;
    }
    Ok(Array::from_slice(&packed, &[n_bytes as i32]))
}

/// Inverse of [`bit_pack`]: recover `n` `u8` codes from a packed `u8` buffer.
///
/// `packed` must be the buffer [`bit_pack`] produced for `n` codes at the same `bits`. Returns a
/// 1-D `u8` [`Array`] of length `n`.
pub fn bit_unpack(packed: &Array, n: usize, bits: i32) -> Result<Array> {
    check_bits(bits)?;
    let expected = packed_len(n, bits)?;
    let packed = packed.as_dtype(Dtype::Uint8)?;
    let buf: Vec<u8> = packed.as_slice::<u8>().to_vec();
    if buf.len() != expected {
        return Err(Error::Msg(format!(
            "bit_unpack: packed length {} != expected {expected} for n={n} bits={bits}",
            buf.len()
        )));
    }
    let per_byte = (8 / bits) as usize;
    let mask: u8 = ((1u16 << bits) - 1) as u8;

    let mut codes = vec![0u8; n];
    // TODO(sc-8529/Phase2): replace this host-side unpack loop with a MetalKernel (one thread/code).
    for (elem_idx, out) in codes.iter_mut().enumerate() {
        let byte_idx = elem_idx / per_byte;
        let bit_off = (elem_idx % per_byte) * bits as usize;
        *out = (buf[byte_idx] >> bit_off as u32) & mask;
    }
    Ok(Array::from_slice(&codes, &[n as i32]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u8s(a: &Array) -> Vec<u8> {
        a.as_dtype(Dtype::Uint8).unwrap().as_slice::<u8>().to_vec()
    }

    /// HAND-COMPUTED 4-bit pack: codes [1, 2, 3, 4] -> bytes:
    ///   byte0 = 1 | (2<<4) = 0x21 = 33
    ///   byte1 = 3 | (4<<4) = 0x43 = 67
    #[test]
    fn pack_4bit_hand_vector() {
        let codes = Array::from_slice(&[1u8, 2, 3, 4], &[4]);
        let packed = bit_pack(&codes, 4).unwrap();
        assert_eq!(u8s(&packed), vec![33, 67]);
        let back = bit_unpack(&packed, 4, 4).unwrap();
        assert_eq!(u8s(&back), vec![1, 2, 3, 4]);
    }

    /// HAND-COMPUTED 2-bit pack: codes [3, 2, 1, 0] ->
    ///   byte0 = 3 | (2<<2) | (1<<4) | (0<<6) = 0b00011011 = 27
    #[test]
    fn pack_2bit_hand_vector() {
        let codes = Array::from_slice(&[3u8, 2, 1, 0], &[4]);
        let packed = bit_pack(&codes, 2).unwrap();
        assert_eq!(u8s(&packed), vec![27]);
        assert_eq!(u8s(&bit_unpack(&packed, 4, 2).unwrap()), vec![3, 2, 1, 0]);
    }

    /// HAND-COMPUTED 1-bit pack: codes [1,0,1,0,1,1,0,0] ->
    ///   byte0 = 1 | (1<<2) | (1<<4) | (1<<5) = 0b00110101 = 53
    #[test]
    fn pack_1bit_hand_vector() {
        let codes = Array::from_slice(&[1u8, 0, 1, 0, 1, 1, 0, 0], &[8]);
        let packed = bit_pack(&codes, 1).unwrap();
        assert_eq!(u8s(&packed), vec![53]);
        let back = bit_unpack(&packed, 8, 1).unwrap();
        assert_eq!(u8s(&back), vec![1, 0, 1, 0, 1, 1, 0, 0]);
    }

    /// Losslessness oracle: unpack(pack(codes)) == codes for every bit-width, including the
    /// all-zero and all-max edge cases and a deterministic pseudo-random fill.
    #[test]
    fn pack_unpack_lossless_all_widths() {
        for &bits in &[1i32, 2, 4] {
            let max = (1u8 << bits) - 1;
            let per_byte = (8 / bits) as usize;
            // Use a length that is a multiple of per_byte and reasonably large.
            let n = per_byte * 257; // 257 prime-ish to avoid accidental alignment
                                    // Edge: all zeros.
            let zeros = vec![0u8; n];
            let za = Array::from_slice(&zeros, &[n as i32]);
            assert_eq!(
                u8s(&bit_unpack(&bit_pack(&za, bits).unwrap(), n, bits).unwrap()),
                zeros
            );
            // Edge: all max.
            let maxs = vec![max; n];
            let ma = Array::from_slice(&maxs, &[n as i32]);
            assert_eq!(
                u8s(&bit_unpack(&bit_pack(&ma, bits).unwrap(), n, bits).unwrap()),
                maxs
            );
            // Deterministic pseudo-random in-range codes (LCG).
            let mut state: u32 = 0x1234_5678;
            let rnd: Vec<u8> = (0..n)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((state >> 24) as u8) & max
                })
                .collect();
            let ra = Array::from_slice(&rnd, &[n as i32]);
            let round = u8s(&bit_unpack(&bit_pack(&ra, bits).unwrap(), n, bits).unwrap());
            assert_eq!(round, rnd, "lossless failed at bits={bits}");
        }
    }

    /// Out-of-range high bits are masked, not bled into the neighbour code.
    #[test]
    fn pack_masks_out_of_range_bits() {
        // 2-bit: code 0xFF should mask to 0x03.
        let codes = Array::from_slice(&[0xFFu8, 0, 0, 0], &[4]);
        let packed = bit_pack(&codes, 2).unwrap();
        // byte0 = 3 | 0 | 0 | 0 = 3
        assert_eq!(u8s(&packed), vec![3]);
        assert_eq!(u8s(&bit_unpack(&packed, 4, 2).unwrap()), vec![3, 0, 0, 0]);
    }

    #[test]
    fn rejects_bad_widths_and_alignment() {
        let codes = Array::from_slice(&[0u8, 1, 2], &[3]);
        assert!(bit_pack(&codes, 3).is_err()); // unsupported width
        assert!(bit_pack(&codes, 4).is_err()); // 3 not divisible by 2
        assert!(packed_len(3, 4).is_err());
        assert_eq!(packed_len(4, 4).unwrap(), 2);
        assert_eq!(packed_len(8, 1).unwrap(), 1);
    }
}
