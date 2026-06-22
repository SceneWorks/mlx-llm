//! Thin FFI over mlx-c's JIT custom Metal kernel API (`mlx_fast_metal_kernel_*`).
//!
//! mlx-rs does not surface custom Metal kernels at the Rust level, but the C API is compiled into the
//! linked mlx-c and `mlx_rs::Array::as_ptr()` exposes the underlying `mlx_array` handle — so the
//! paged-attention kernel (story 7170) reaches it directly via the `mlx-sys` bindings, with **no fork
//! patch or MLX-core rebuild**. This module is the small unsafe boundary; everything above it is safe.
//!
//! A [`MetalKernel`] JIT-compiles a Metal source once (MLX caches by name+source); [`MetalKernel::run`]
//! dispatches it for one output. Single-threaded, like the rest of the engine (one MLX device).

use std::ffi::CString;

use mlx_rs::{Array, Dtype};

use crate::error::{Error, Result};

/// A compile-time template argument substituted into the kernel source by MLX.
#[derive(Clone, Copy, Debug)]
pub enum TemplateArg<'a> {
    /// `template <typename name>` bound to an MLX dtype.
    Dtype(&'a str, Dtype),
    /// An `int` template constant.
    Int(&'a str, i32),
    /// A `bool` template constant.
    Bool(&'a str, bool),
}

/// A JIT-compiled custom Metal kernel.
pub struct MetalKernel {
    inner: mlx_sys::mlx_fast_metal_kernel,
}

impl MetalKernel {
    /// Compile a kernel `name` with the given input/output parameter names and Metal `source`. Inputs
    /// are addressable in the source by their names; the single output by its name. `source` is the
    /// kernel **body** (MLX wraps it in the function signature).
    pub fn new(name: &str, input_names: &[&str], output_names: &[&str], source: &str) -> Result<Self> {
        Self::new_with_header(name, input_names, output_names, "", source)
    }

    /// Like [`new`](Self::new) but with a file-scope `header` (e.g. `#include <metal_simdgroup>`)
    /// emitted before the generated function signature — needed for kernels using simd-reduction
    /// intrinsics (`simd_sum`/`simd_max`).
    pub fn new_with_header(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        header: &str,
        source: &str,
    ) -> Result<Self> {
        let c_name = cstr(name)?;
        let c_source = cstr(source)?;
        let c_header = cstr(header)?;
        let inputs = VectorString::new(input_names)?;
        let outputs = VectorString::new(output_names)?;
        // ensure_row_contiguous = true (we pass plain contiguous arrays); atomic_outputs = false.
        let inner = unsafe {
            mlx_sys::mlx_fast_metal_kernel_new(
                c_name.as_ptr(),
                inputs.0,
                outputs.0,
                c_source.as_ptr(),
                c_header.as_ptr(),
                true,
                false,
            )
        };
        if inner.ctx.is_null() {
            return Err(Error::Msg(format!("mlx_fast_metal_kernel_new failed for '{name}'")));
        }
        Ok(Self { inner })
    }

    /// Dispatch the kernel producing one output of `output_shape`/`output_dtype`. `grid` is the total
    /// thread count per axis; `threadgroup` the threads per threadgroup. `template` binds the source's
    /// template parameters. Inputs are passed in declaration order.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        inputs: &[&Array],
        output_shape: &[i32],
        output_dtype: Dtype,
        grid: (i32, i32, i32),
        threadgroup: (i32, i32, i32),
        template: &[TemplateArg<'_>],
    ) -> Result<Array> {
        let in_vec = VectorArray::new(inputs);
        let config = Config::new()?;
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_set_grid(config.0, grid.0, grid.1, grid.2);
            mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(
                config.0,
                threadgroup.0,
                threadgroup.1,
                threadgroup.2,
            );
            mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                config.0,
                output_shape.as_ptr(),
                output_shape.len(),
                dtype_to_c(output_dtype),
            );
        }
        // Hold the template-arg names alive across the FFI calls.
        let names: Vec<CString> = template
            .iter()
            .map(|t| match t {
                TemplateArg::Dtype(n, _) | TemplateArg::Int(n, _) | TemplateArg::Bool(n, _) => cstr(n),
            })
            .collect::<Result<_>>()?;
        for (arg, name) in template.iter().zip(&names) {
            unsafe {
                match arg {
                    TemplateArg::Dtype(_, d) => mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                        config.0,
                        name.as_ptr(),
                        dtype_to_c(*d),
                    ),
                    TemplateArg::Int(_, v) => mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                        config.0,
                        name.as_ptr(),
                        *v,
                    ),
                    TemplateArg::Bool(_, v) => mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
                        config.0,
                        name.as_ptr(),
                        *v,
                    ),
                };
            }
        }

        let stream = Stream::default_gpu();
        let mut out_vec = unsafe { mlx_sys::mlx_vector_array_new() };
        let status = unsafe {
            mlx_sys::mlx_fast_metal_kernel_apply(&mut out_vec, self.inner, in_vec.0, config.0, stream.0)
        };
        if status != 0 {
            unsafe { mlx_sys::mlx_vector_array_free(out_vec) };
            return Err(Error::Msg("mlx_fast_metal_kernel_apply failed".into()));
        }
        let mut out = unsafe { mlx_sys::mlx_array_new() };
        unsafe {
            mlx_sys::mlx_vector_array_get(&mut out, out_vec, 0);
            mlx_sys::mlx_vector_array_free(out_vec);
        }
        // `out` is now an owned mlx_array reference; hand it to mlx-rs (it frees on drop).
        Ok(unsafe { Array::from_ptr(out) })
    }
}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_fast_metal_kernel_free(self.inner) };
    }
}

// ---- small RAII wrappers around the mlx-c handles ----

struct Config(mlx_sys::mlx_fast_metal_kernel_config);
impl Config {
    fn new() -> Result<Self> {
        let c = unsafe { mlx_sys::mlx_fast_metal_kernel_config_new() };
        if c.ctx.is_null() {
            return Err(Error::Msg("mlx_fast_metal_kernel_config_new failed".into()));
        }
        Ok(Self(c))
    }
}
impl Drop for Config {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_fast_metal_kernel_config_free(self.0) };
    }
}

struct VectorString(mlx_sys::mlx_vector_string);
impl VectorString {
    fn new(names: &[&str]) -> Result<Self> {
        let v = unsafe { mlx_sys::mlx_vector_string_new() };
        for n in names {
            let c = cstr(n)?;
            unsafe { mlx_sys::mlx_vector_string_append_value(v, c.as_ptr()) };
        }
        Ok(Self(v))
    }
}
impl Drop for VectorString {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_vector_string_free(self.0) };
    }
}

struct VectorArray(mlx_sys::mlx_vector_array);
impl VectorArray {
    fn new(arrays: &[&Array]) -> Self {
        let v = unsafe { mlx_sys::mlx_vector_array_new() };
        for a in arrays {
            unsafe { mlx_sys::mlx_vector_array_append_value(v, a.as_ptr()) };
        }
        Self(v)
    }
}
impl Drop for VectorArray {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_vector_array_free(self.0) };
    }
}

struct Stream(mlx_sys::mlx_stream);
impl Stream {
    fn default_gpu() -> Self {
        Self(unsafe { mlx_sys::mlx_default_gpu_stream_new() })
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_stream_free(self.0) };
    }
}

fn cstr(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error::Msg("kernel string contains a NUL byte".into()))
}

fn dtype_to_c(d: Dtype) -> mlx_sys::mlx_dtype {
    match d {
        Dtype::Float32 => mlx_sys::mlx_dtype__MLX_FLOAT32,
        Dtype::Bfloat16 => mlx_sys::mlx_dtype__MLX_BFLOAT16,
        Dtype::Int32 => mlx_sys::mlx_dtype__MLX_INT32,
        Dtype::Uint32 => mlx_sys::mlx_dtype__MLX_UINT32,
        // The paged kernel uses only the dtypes above; extend as needed.
        other => panic!("dtype_to_c: unsupported dtype {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elementwise_exp_kernel_runs() {
        // The mlx-c worked example: out[e] = exp(inp[e]). Proves the FFI + linking + JIT path.
        let src = "uint e = thread_position_in_grid.x; T t = inp[e]; out[e] = metal::exp(t);";
        let kernel = MetalKernel::new("smoke_exp", &["inp"], &["out"], src).unwrap();
        let input = Array::from_slice(&[0.0f32, 1.0, 2.0, -1.0], &[4]);
        let out = kernel
            .run(
                &[&input],
                &[4],
                Dtype::Float32,
                (4, 1, 1),
                (4, 1, 1),
                &[TemplateArg::Dtype("T", Dtype::Float32)],
            )
            .unwrap();
        let got = out.as_slice::<f32>().to_vec();
        let want = [0.0f32, 1.0, 2.0, -1.0].map(f32::exp);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "{g} vs {w}");
        }
    }
}
