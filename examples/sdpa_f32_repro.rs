//! Standalone repro for **sc-7430**: localize the wrong-numbers bug on the pinned pmetal mlx-rs fork.
//!
//! The story's hypothesis was that BOTH MLX's fused `scaled_dot_product_attention` AND a raw **4-D
//! batched `matmul`** return wrong **f32** results at multi-head / `seq >= 16` / `head_dim >= 64`
//! shapes. This repro tests that hypothesis from scratch using **only `mlx_rs::*`** (no mlx-llm
//! internals), against a pure-Rust **f64** host reference (the ground truth).
//!
//! Run (release, to match the engine's build incl. `MACOSX_DEPLOYMENT_TARGET=26.2`):
//!   cargo run --release --example sdpa_f32_repro
//!
//! Two sections:
//!
//!   SECTION A — matmul primitive. Raw QKᵀ via a 4-D batched `matmul` on GPU (`mm4_gpu`), the SAME
//!   op on the CPU stream (`mm4_cpu`), and the 3-D-folded form the engine uses as a "workaround"
//!   (`mm3_gpu`). If the story were right, `mm4_gpu` would diverge while `mm3_gpu` stayed correct.
//!
//!   SECTION B — attention. Fused SDPA on GPU (`sdpa_gpu`) and CPU (`sdpa_cpu`), plus a hand-rolled
//!   eager attention built from raw **4-D** matmul + softmax + matmul on GPU (`manual4d`). Swept over
//!   BOTH mask modes (none / causal) and over the real decode shape (q_len=1, k_len large) as well as
//!   prefill (q_len=k_len). `sdpa_cpu` vs `sdpa_gpu` localizes a Metal-kernel bug; `manual4d` vs
//!   `sdpa_gpu` shows whether the raw-matmul path is the correct one.
//!
//! Methodology: inputs are rounded to the test dtype, then read back to f32 and fed to the host
//! reference, so a reported error is *kernel* error, not input-rounding error.

#![allow(clippy::too_many_arguments)] // the diagnostic sweep helpers take many axis params by design

use mlx_rs::fast::{scaled_dot_product_attention_device, ScaledDotProductAttentionMask};
use mlx_rs::ops::{add, matmul_device, multiply, softmax_axis};
use mlx_rs::{Array, Dtype, StreamOrDevice};

const MASK_NEG: f32 = -1e30;

/// Deterministic uniform `[-1, 1]` data — the same LCG the attention unit tests use.
fn randf(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Cast an f32 host buffer to `dt` as an MLX array AND return the dtype-rounded values read back to
/// f32 — the host reference consumes the rounded values so we measure kernel error, not rounding.
fn to_dtype(data: &[f32], shape: &[i32], dt: Dtype) -> (Array, Vec<f32>) {
    let a = Array::from_slice(data, shape).as_dtype(dt).unwrap();
    let rounded = a.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec();
    (a, rounded)
}

/// Read an MLX array back as f32 regardless of stored dtype (forces evaluation).
fn read_f32(a: &Array) -> Vec<f32> {
    a.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec()
}

/// Relative max error: `max|mlx − host| / (max|host| + ε)`. Scale-free so dtypes are comparable.
fn rel_err(mlx: &[f32], host: &[f32]) -> f32 {
    let max_abs = mlx
        .iter()
        .zip(host)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_host = host.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    max_abs / (max_host + 1e-20)
}

/// Per-dtype FAIL threshold: above correct-kernel rounding noise, far below the bug's rel ≈ 0.5+.
fn fail_threshold(dt: Dtype) -> f32 {
    match dt {
        Dtype::Float32 => 2e-3,
        _ => 8e-2,
    }
}

fn cell(v: f32, dt: Dtype) -> String {
    if v.is_nan() {
        return format!("{:>10}", "n/a");
    }
    let flag = if v > fail_threshold(dt) { "*" } else { " " };
    format!("{:>9.2e}{}", v, flag)
}

// ===== Host references (pure Rust, f64 accumulation) ==========================================

/// QKᵀ scores (no scale, no mask). `q`/`k` are `[h, s, hd]` row-major (b folded into h by caller).
fn host_scores(q: &[f32], k: &[f32], bh: usize, s: usize, hd: usize) -> Vec<f32> {
    let mut out = vec![0f32; bh * s * s];
    for g in 0..bh {
        let base = g * s * hd;
        for i in 0..s {
            for j in 0..s {
                let mut acc = 0f64;
                for d in 0..hd {
                    acc += q[base + i * hd + d] as f64 * k[base + j * hd + d] as f64;
                }
                out[g * s * s + i * s + j] = acc as f32;
            }
        }
    }
    out
}

/// `softmax(scale · QKᵀ [+ causal]) · V`, asymmetric q/k lengths, f64 accumulation. Causal aligns
/// q to the bottom-right of k (offset = kl − ql), matching MLX's implicit-causal convention.
fn host_attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bh: usize,
    ql: usize,
    kl: usize,
    hd: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let offset = kl as isize - ql as isize;
    let mut out = vec![0f32; bh * ql * hd];
    for g in 0..bh {
        let qb = g * ql * hd;
        let kb = g * kl * hd;
        for i in 0..ql {
            let jmax = if causal {
                (offset + i as isize) as usize
            } else {
                kl - 1
            };
            let mut logits = vec![0f64; kl];
            let mut m = f64::MIN;
            for j in 0..=jmax {
                let mut acc = 0f64;
                for d in 0..hd {
                    acc += q[qb + i * hd + d] as f64 * k[kb + j * hd + d] as f64;
                }
                logits[j] = acc * scale as f64;
                m = m.max(logits[j]);
            }
            let mut denom = 0f64;
            for lj in logits.iter_mut().take(jmax + 1) {
                *lj = (*lj - m).exp();
                denom += *lj;
            }
            for d in 0..hd {
                let mut acc = 0f64;
                for j in 0..=jmax {
                    acc += logits[j] / denom * v[kb + j * hd + d] as f64;
                }
                out[qb + i * hd + d] = acc as f32;
            }
        }
    }
    out
}

// ===== Section A: matmul primitive ===========================================================

struct MmRow {
    dt_name: &'static str,
    dt: Dtype,
    b: usize,
    h: usize,
    s: usize,
    hd: usize,
    mm4_gpu: f32,
    mm4_cpu: f32,
    mm3_gpu: f32,
}

fn run_matmul(dt_name: &'static str, dt: Dtype, b: usize, h: usize, s: usize, hd: usize) -> MmRow {
    let n = b * h * s * hd;
    let sh = [b as i32, h as i32, s as i32, hd as i32];
    let (q, qr) = to_dtype(&randf(n, 1), &sh, dt);
    let (k, kr) = to_dtype(&randf(n, 2), &sh, dt);
    let host = host_scores(&qr, &kr, b * h, s, hd);
    let kt = k.transpose_axes(&[0, 1, 3, 2]).unwrap();

    let mm4_gpu = matmul_device(&q, &kt, StreamOrDevice::gpu()).unwrap();
    let mm4_cpu = matmul_device(&q, &kt, StreamOrDevice::cpu()).unwrap();
    let q3 = q.reshape(&[(b * h) as i32, s as i32, hd as i32]).unwrap();
    let kt3 = kt.reshape(&[(b * h) as i32, hd as i32, s as i32]).unwrap();
    let mm3_gpu = matmul_device(&q3, &kt3, StreamOrDevice::gpu()).unwrap();

    MmRow {
        dt_name,
        dt,
        b,
        h,
        s,
        hd,
        mm4_gpu: rel_err(&read_f32(&mm4_gpu), &host),
        mm4_cpu: rel_err(&read_f32(&mm4_cpu), &host),
        mm3_gpu: rel_err(&read_f32(&mm3_gpu), &host),
    }
}

// ===== Section B: attention ==================================================================

struct AttnRow {
    dt_name: &'static str,
    dt: Dtype,
    h: usize,
    ql: usize,
    kl: usize,
    hd: usize,
    causal: bool,
    sdpa_gpu: f32,
    sdpa_cpu: f32,
    manual4d: f32,
}

fn run_attn(dt_name: &'static str, dt: Dtype, h: usize, ql: usize, kl: usize, hd: usize, causal: bool) -> AttnRow {
    let scale = 1.0 / (hd as f32).sqrt();
    let qsh = [1, h as i32, ql as i32, hd as i32];
    let ksh = [1, h as i32, kl as i32, hd as i32];
    let (q, qr) = to_dtype(&randf(h * ql * hd, 1), &qsh, dt);
    let (k, kr) = to_dtype(&randf(h * kl * hd, 2), &ksh, dt);
    let (v, vr) = to_dtype(&randf(h * kl * hd, 3), &ksh, dt);
    let host = host_attn(&qr, &kr, &vr, h, ql, kl, hd, scale, causal);

    let mask_arg = if causal {
        Some(ScaledDotProductAttentionMask::Causal)
    } else {
        None
    };
    let sdpa_gpu = scaled_dot_product_attention_device(
        &q, &k, &v, scale, mask_arg, None::<&Array>, StreamOrDevice::gpu(),
    )
    .map(|a| rel_err(&read_f32(&a), &host))
    .unwrap_or(f32::NAN);
    let mask_arg_cpu = if causal {
        Some(ScaledDotProductAttentionMask::Causal)
    } else {
        None
    };
    let sdpa_cpu = scaled_dot_product_attention_device(
        &q, &k, &v, scale, mask_arg_cpu, None::<&Array>, StreamOrDevice::cpu(),
    )
    .map(|a| rel_err(&read_f32(&a), &host))
    .unwrap_or(f32::NAN);

    // Hand-rolled eager attention via raw 4-D matmul + softmax + 4-D matmul (all on GPU, in dtype).
    let kt = k.transpose_axes(&[0, 1, 3, 2]).unwrap();
    let mut scores = multiply(
        matmul_device(&q, &kt, StreamOrDevice::gpu()).unwrap(),
        Array::from_f32(scale),
    )
    .unwrap();
    if causal {
        let offset = kl as isize - ql as isize;
        let mut md = vec![0f32; ql * kl];
        for i in 0..ql {
            for j in 0..kl {
                if (j as isize) > offset + i as isize {
                    md[i * kl + j] = MASK_NEG;
                }
            }
        }
        let mask = Array::from_slice(&md, &[1, 1, ql as i32, kl as i32])
            .as_dtype(dt)
            .unwrap();
        scores = add(&scores, &mask).unwrap();
    }
    let axis = scores.ndim() as i32 - 1;
    let weights = softmax_axis(&scores, axis, None).unwrap();
    let out = matmul_device(&weights, &v, StreamOrDevice::gpu()).unwrap();
    let manual4d = rel_err(&read_f32(&out), &host);

    AttnRow {
        dt_name,
        dt,
        h,
        ql,
        kl,
        hd,
        causal,
        sdpa_gpu,
        sdpa_cpu,
        manual4d,
    }
}

/// Like [`run_attn`] but scales the uniform inputs by `amp` (and optionally makes Q≈K-aligned to
/// force a *peaked* softmax) — to test whether the fused-SDPA bug is data-dependent (i.e. whether
/// the structured/peaked distributions real weights produce avoid it) or structural (hits any data).
/// Returns the GPU fused-SDPA relative error vs the f64 host reference.
fn run_attn_amp(dt: Dtype, h: usize, ql: usize, kl: usize, hd: usize, causal: bool, amp: f32, peaked: bool) -> f32 {
    let scale = 1.0 / (hd as f32).sqrt();
    let qsh = [1, h as i32, ql as i32, hd as i32];
    let ksh = [1, h as i32, kl as i32, hd as i32];
    let mut qd: Vec<f32> = randf(h * ql * hd, 1).iter().map(|x| x * amp).collect();
    let kd: Vec<f32> = randf(h * kl * hd, 2).iter().map(|x| x * amp).collect();
    let vd: Vec<f32> = randf(h * kl * hd, 3).iter().map(|x| x * amp).collect();
    if peaked {
        // Align each query with one key so the softmax concentrates (the real-attention regime).
        for g in 0..h {
            for i in 0..ql {
                let j = i % kl;
                for d in 0..hd {
                    qd[(g * ql + i) * hd + d] = kd[(g * kl + j) * hd + d] * 4.0;
                }
            }
        }
    }
    let (q, qr) = to_dtype(&qd, &qsh, dt);
    let (k, kr) = to_dtype(&kd, &ksh, dt);
    let (v, vr) = to_dtype(&vd, &ksh, dt);
    let host = host_attn(&qr, &kr, &vr, h, ql, kl, hd, scale, causal);
    let mask_arg = if causal { Some(ScaledDotProductAttentionMask::Causal) } else { None };
    scaled_dot_product_attention_device(&q, &k, &v, scale, mask_arg, None::<&Array>, StreamOrDevice::gpu())
        .map(|a| rel_err(&read_f32(&a), &host))
        .unwrap_or(f32::NAN)
}

/// Expand GQA KV `[kvh, len, hd]` (logical row-major) to `[kvh*groups, len, hd]` — each kv head
/// repeated `groups` times adjacently, matching `repeat_kv` and the fused kernel's gqa convention.
fn expand_kv(data: &[f32], kvh: usize, len: usize, hd: usize, groups: usize) -> Vec<f32> {
    let mut out = vec![0f32; kvh * groups * len * hd];
    for kh in 0..kvh {
        for g in 0..groups {
            let dst = (kh * groups + g) * len * hd;
            let src = kh * len * hd;
            out[dst..dst + len * hd].copy_from_slice(&data[src..src + len * hd]);
        }
    }
    out
}

/// Build a `[1, heads, len, hd]` MLX array from `logical` (row-major `[heads, len, hd]`). When
/// `strided`, the array is a transposed view of `[1, len, heads, hd]` storage — exactly the layout
/// the model hands SDPA (`apply_rope(...).transpose_axes([0,2,1,3])`). Returns the array plus its
/// dtype-rounded logical values for the host reference.
fn mk(logical: &[f32], heads: usize, len: usize, hd: usize, dt: Dtype, strided: bool) -> (Array, Vec<f32>) {
    let a = if strided {
        let mut st = vec![0f32; len * heads * hd];
        for g in 0..heads {
            for i in 0..len {
                for d in 0..hd {
                    st[(i * heads + g) * hd + d] = logical[(g * len + i) * hd + d];
                }
            }
        }
        Array::from_slice(&st, &[1, len as i32, heads as i32, hd as i32])
            .transpose_axes(&[0, 2, 1, 3])
            .unwrap()
            .as_dtype(dt)
            .unwrap()
    } else {
        Array::from_slice(logical, &[1, heads as i32, len as i32, hd as i32])
            .as_dtype(dt)
            .unwrap()
    };
    let rounded = read_f32(&a); // logical [heads, len, hd] order
    (a, rounded)
}

/// Fused-SDPA-GPU rel error for an arbitrary GQA config and input layout — used to bisect the real
/// trigger (MHA vs GQA, contiguous vs strided) at the engine's actual prefill shape.
fn run_attn_gqa(dt: Dtype, qh: usize, kvh: usize, ql: usize, kl: usize, hd: usize, causal: bool, strided: bool) -> f32 {
    let groups = qh / kvh;
    let scale = 1.0 / (hd as f32).sqrt();
    let (q, qr) = mk(&randf(qh * ql * hd, 1), qh, ql, hd, dt, strided);
    let (k, kr) = mk(&randf(kvh * kl * hd, 2), kvh, kl, hd, dt, strided);
    let (v, vr) = mk(&randf(kvh * kl * hd, 3), kvh, kl, hd, dt, strided);
    let kx = expand_kv(&kr, kvh, kl, hd, groups);
    let vx = expand_kv(&vr, kvh, kl, hd, groups);
    let host = host_attn(&qr, &kx, &vx, qh, ql, kl, hd, scale, causal);
    let mask_arg = if causal { Some(ScaledDotProductAttentionMask::Causal) } else { None };
    scaled_dot_product_attention_device(&q, &k, &v, scale, mask_arg, None::<&Array>, StreamOrDevice::gpu())
        .map(|a| rel_err(&read_f32(&a), &host))
        .unwrap_or(f32::NAN)
}

fn main() {
    let dtypes = [
        ("f32", Dtype::Float32),
        ("bf16", Dtype::Bfloat16),
        ("f16", Dtype::Float16),
    ];

    println!("sc-7430 repro — pinned pmetal mlx-rs fork; uniform[-1,1]; '*' = rel error over FAIL threshold");
    println!("(f32 FAIL > 2e-3, bf16/f16 FAIL > 8e-2)\n");

    // ---- Section A: matmul primitive --------------------------------------------------------
    println!("== SECTION A: raw QKᵀ matmul vs f64 host (is the 4-D batched matmul broken?) ==");
    println!(
        "{:>5} {:>2} {:>2} {:>3} {:>4} | {:>10} {:>10} {:>10}",
        "dt", "b", "h", "s", "hd", "mm4_gpu", "mm4_cpu", "mm3_gpu"
    );
    println!("{}", "-".repeat(60));
    let mut mm_fail = 0;
    for (name, dt) in dtypes {
        for &(b, h, s, hd) in &[
            (1usize, 8usize, 64usize, 128usize),
            (1, 2, 16, 64),
            (1, 8, 16, 64),
            (2, 8, 16, 64), // two genuine leading dims
            (2, 2, 64, 128),
        ] {
            let r = run_matmul(name, dt, b, h, s, hd);
            for v in [r.mm4_gpu, r.mm4_cpu, r.mm3_gpu] {
                if v > fail_threshold(dt) {
                    mm_fail += 1;
                }
            }
            println!(
                "{:>5} {:>2} {:>2} {:>3} {:>4} | {} {} {}",
                r.dt_name, r.b, r.h, r.s, r.hd, cell(r.mm4_gpu, r.dt), cell(r.mm4_cpu, r.dt), cell(r.mm3_gpu, r.dt)
            );
        }
    }
    println!(
        "  => matmul failures: {} (mm4_gpu = raw 4-D batched matmul on GPU)\n",
        mm_fail
    );

    // ---- Section B: attention ---------------------------------------------------------------
    println!("== SECTION B: attention vs f64 host (decode q=1/large-k, prefill q=k; none + causal) ==");
    println!(
        "{:>5} {:>2} {:>3} {:>3} {:>4} {:>7} | {:>10} {:>10} {:>10}",
        "dt", "h", "ql", "kl", "hd", "mask", "sdpa_gpu", "sdpa_cpu", "manual4d"
    );
    println!("{}", "-".repeat(74));
    // (h, ql, kl, hd): decode hot path (ql=1) + prefill (ql=kl) + chunked prefill into cache.
    let shapes = [
        (2usize, 1usize, 64usize, 64usize), // decode, multi-head
        (8, 1, 64, 128),                    // decode, Qwen3-like hd=128
        (8, 1, 256, 128),                   // decode, longer cache
        (2, 16, 16, 64),                    // prefill square (the known-bad no-mask shape)
        (8, 64, 64, 128),                   // prefill square, big
        (8, 16, 64, 64),                    // chunked prefill into a 64-key cache
    ];
    let mut hotpath_fail: Vec<String> = Vec::new();
    for (name, dt) in dtypes {
        for &(h, ql, kl, hd) in &shapes {
            for causal in [false, true] {
                let r = run_attn(name, dt, h, ql, kl, hd, causal);
                println!(
                    "{:>5} {:>2} {:>3} {:>3} {:>4} {:>7} | {} {} {}",
                    r.dt_name,
                    r.h,
                    r.ql,
                    r.kl,
                    r.hd,
                    if r.causal { "causal" } else { "none" },
                    cell(r.sdpa_gpu, r.dt),
                    cell(r.sdpa_cpu, r.dt),
                    cell(r.manual4d, r.dt),
                );
                // Real hot path = causal SDPA on GPU (decode + prefill).
                if r.causal && r.sdpa_gpu > fail_threshold(r.dt) {
                    hotpath_fail.push(format!(
                        "{}[h{} ql{} kl{} hd{}]",
                        r.dt_name, r.h, r.ql, r.kl, r.hd
                    ));
                }
            }
        }
    }

    println!("\nCAUSAL fused-SDPA-on-GPU failures (the real decode + prefill hot path):");
    if hotpath_fail.is_empty() {
        println!("  (none — the causal fused SDPA hot path is correct across this sweep)");
    } else {
        println!("  {}", hotpath_fail.join(", "));
    }

    // ---- Section C: q_len threshold (where does the prefill kernel start failing?) -----------
    println!("\n== SECTION C: fused-SDPA-GPU rel error vs q_len (h=8, kl=ql, hd=64, causal) ==");
    print!("{:>5} |", "dt");
    for ql in [1usize, 2, 4, 8, 9, 12, 16, 32] {
        print!("{:>8}", format!("q{ql}"));
    }
    println!();
    println!("{}", "-".repeat(72));
    for (name, dt) in dtypes {
        print!("{:>5} |", name);
        for ql in [1usize, 2, 4, 8, 9, 12, 16, 32] {
            let e = run_attn_amp(dt, 8, ql, ql, 64, true, 1.0, false);
            let flag = if e > fail_threshold(dt) { "*" } else { " " };
            print!("{:>7.1e}{}", e, flag);
        }
        println!();
    }

    // ---- Section E: GQA vs MHA × contiguous vs strided, at the engine's real prefill shape -----
    // SmolLM2-135M prefill = q9/kv3 (GQA), strided inputs, ql=kl=26, hd=64 — and that WORKS in the
    // real model. So which dimension flips the broken MHA-contiguous case to correct?
    println!("\n== SECTION E: real prefill shape (qh=9 ql=26 kl=26 hd=64 causal); MHA/GQA × layout ==");
    println!("{:>5} | {:>14} {:>14} {:>14} {:>16}", "dt", "MHA contig", "MHA strided", "GQA contig", "GQA strided(real)");
    println!("{}", "-".repeat(72));
    for (name, dt) in dtypes {
        let mha_c = run_attn_gqa(dt, 9, 9, 26, 26, 64, true, false);
        let mha_s = run_attn_gqa(dt, 9, 9, 26, 26, 64, true, true);
        let gqa_c = run_attn_gqa(dt, 9, 3, 26, 26, 64, true, false);
        let gqa_s = run_attn_gqa(dt, 9, 3, 26, 26, 64, true, true);
        let f = |v: f32| if v > fail_threshold(dt) { "*" } else { " " };
        println!(
            "{:>5} | {:>13.2e}{} {:>13.2e}{} {:>13.2e}{} {:>15.2e}{}",
            name, mha_c, f(mha_c), mha_s, f(mha_s), gqa_c, f(gqa_c), gqa_s, f(gqa_s)
        );
    }

    // ---- Section F: is q_len<=8 safe even with a large cache? (chunked-prefill mitigation) ------
    println!("\n== SECTION F: fused-SDPA-GPU rel error, q_len<=8 vs large k_len (h=8 hd=64 causal) ==");
    print!("{:>5} |", "dt");
    for &(ql, kl) in &[(2usize, 64usize), (4, 64), (8, 64), (8, 256), (2, 256)] {
        print!("{:>11}", format!("q{ql}/k{kl}"));
    }
    println!();
    println!("{}", "-".repeat(64));
    for (name, dt) in dtypes {
        print!("{:>5} |", name);
        for &(ql, kl) in &[(2usize, 64usize), (4, 64), (8, 64), (8, 256), (2, 256)] {
            let e = run_attn_amp(dt, 8, ql, kl, 64, true, 1.0, false);
            let flag = if e > fail_threshold(dt) { "*" } else { " " };
            print!("{:>10.1e}{}", e, flag);
        }
        println!();
    }

    // ---- Section G: head_dim threshold (h=8, ql=16, kl=16, causal) -------------------------------
    println!("\n== SECTION G: fused-SDPA-GPU rel error vs head_dim (h=8 ql16 kl16 causal) ==");
    print!("{:>5} |", "dt");
    for hd in [16usize, 32, 48, 64, 96, 128] {
        print!("{:>9}", format!("hd{hd}"));
    }
    println!();
    println!("{}", "-".repeat(64));
    for (name, dt) in dtypes {
        print!("{:>5} |", name);
        for hd in [16usize, 32, 48, 64, 96, 128] {
            let e = run_attn_amp(dt, 8, 16, 16, hd, true, 1.0, false);
            let flag = if e > fail_threshold(dt) { "*" } else { " " };
            print!("{:>8.1e}{}", e, flag);
        }
        println!();
    }

    // ---- Section D: data dependence (does the bug avoid the structured data real weights make?) -
    println!("\n== SECTION D: is the bug data-dependent? fused-SDPA-GPU rel error, h=8 ql16 kl16 hd64 causal ==");
    println!("{:>5} | {:>12} {:>12} {:>12} {:>14}", "dt", "amp=1.0", "amp=0.1", "amp=0.02", "peaked(amp1)");
    println!("{}", "-".repeat(64));
    for (name, dt) in dtypes {
        let a = run_attn_amp(dt, 8, 16, 16, 64, true, 1.0, false);
        let b = run_attn_amp(dt, 8, 16, 16, 64, true, 0.1, false);
        let c = run_attn_amp(dt, 8, 16, 16, 64, true, 0.02, false);
        let p = run_attn_amp(dt, 8, 16, 16, 64, true, 1.0, true);
        let f = |v: f32| if v > fail_threshold(dt) { "*" } else { " " };
        println!(
            "{:>5} | {:>11.2e}{} {:>11.2e}{} {:>11.2e}{} {:>13.2e}{}",
            name, a, f(a), b, f(b), c, f(c), p, f(p)
        );
    }
}
