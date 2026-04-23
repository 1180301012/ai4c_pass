import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_R": 32, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_R": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_R": 64, "BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_R": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_R": 128, "BLOCK_K": 64}, num_warps=8),
        triton.Config({"BLOCK_R": 256, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["R", "K"],
)
@triton.jit
def _fused_scaled_add_transpose_kernel(
    padded_ptr,
    factor_ptr,
    out_ptr,
    R,
    K,
    scale,
    p_stride_h,
    p_stride_r,
    p_stride_k,
    f_stride_h,
    f_stride_r,
    f_stride_k,
    o_stride_r,
    o_stride_h,
    o_stride_k,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    head = pid1
    r_offsets = pid0 * BLOCK_R + tl.arange(0, BLOCK_R)
    k_offsets = tl.arange(0, BLOCK_K)

    mask = (r_offsets[:, None] < R) & (k_offsets[None, :] < K)

    p_ptrs = padded_ptr + head * p_stride_h + r_offsets[:, None] * p_stride_r + k_offsets[None, :] * p_stride_k
    f_ptrs = factor_ptr + head * f_stride_h + r_offsets[:, None] * f_stride_r + k_offsets[None, :] * f_stride_k

    p_vals = tl.load(p_ptrs, mask=mask, other=0.0)
    f_vals = tl.load(f_ptrs, mask=mask, other=0.0)
    out_vals = p_vals + f_vals * scale

    # output shape is [1, R, 8, K], equal to tmp_10 and contiguous in this layout
    o_ptrs = out_ptr + r_offsets[:, None] * o_stride_r + head * o_stride_h + k_offsets[None, :] * o_stride_k
    tl.store(o_ptrs, out_vals, mask=mask)


@torch.fx.wrap
def fused_tail_dispatch(padded, factor, scale, route):
    # padded is tmp_7 with shape [1, 8, R, K]
    # factor is in_4 with shape [1, 8, R, K]
    # return tmp_10 with shape [1, R, 8, K] and contiguous layout so the next reshape is a view.
    R = padded.shape[2]
    K = padded.shape[3]
    out = torch.empty((1, R, 8, K), device=padded.device, dtype=padded.dtype)

    grid = lambda meta: (triton.cdiv(R, meta["BLOCK_R"]), 8)
    _fused_scaled_add_transpose_kernel[grid](
        padded,
        factor,
        out,
        R,
        K,
        float(scale),
        padded.stride(1),
        padded.stride(2),
        padded.stride(3),
        factor.stride(1),
        factor.stride(2),
        factor.stride(3),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )
    return out