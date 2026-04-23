import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_3 = x.transpose(-2, -1)
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return tmp_4


def replacement_args(x):
    return (x,)


@triton.jit
def _transpose_gelu_kernel_bp(
    x_ptr,
    out_ptr,
    n_batch,
    n_time,
    n_cols,
    stride_xb,
    stride_xt,
    stride_xc,
    stride_ob,
    stride_oc,
    stride_ot,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr + pid_b * stride_xb,
        shape=(n_time, n_cols),
        strides=(stride_xt, stride_xc),
        offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
        block_shape=(BLOCK_T, BLOCK_C),
        order=(1, 0),
    )
    x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    inv_sqrt2 = 0.70710678118654752440
    y = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    out_block_ptr = tl.make_block_ptr(
        base=out_ptr + pid_b * stride_ob,
        shape=(n_cols, n_time),
        strides=(stride_oc, stride_ot),
        offsets=(pid_c * BLOCK_C, pid_t * BLOCK_T),
        block_shape=(BLOCK_C, BLOCK_T),
        order=(1, 0),
    )
    tl.store(out_block_ptr, tl.trans(y).to(out_ptr.dtype.element_ty), boundary_check=(0, 1))


@torch.fx.wrap
def fused_transpose_gelu_2d_block_ptr(x):
    n_batch = x.shape[0]
    n_time = x.shape[-2]
    n_cols = x.shape[-1]

    out = torch.empty((n_batch, n_cols, n_time), device=x.device, dtype=x.dtype)

    BLOCK_T = 64
    BLOCK_C = 128
    grid = (
        triton.cdiv(n_time, BLOCK_T),
        triton.cdiv(n_cols, BLOCK_C),
        n_batch,
    )

    _transpose_gelu_kernel_bp[grid](
        x,
        out,
        n_batch,
        n_time,
        n_cols,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_T=BLOCK_T,
        BLOCK_C=BLOCK_C,
        num_warps=8,
        num_stages=2,
    )

    return out


def replacement_func():
    return fused_transpose_gelu_2d_block_ptr