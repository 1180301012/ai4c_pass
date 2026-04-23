import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    tmp_4 = torch.select(tmp_2, 2, 0)
    tmp_5 = torch.select(tmp_2, 2, 1)
    tmp_6 = tmp_5.permute(0, 2, 1)
    return (tmp_6, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 16}, num_warps=2),
        triton.Config({'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_K': 128}, num_warps=8),
    ],
    key=[],
)
@triton.jit
def fused_mul_add_unbind_permute_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, out0_ptr, out1_ptr,
    B, M, N,
    in_0_stride0, in_0_stride1,
    in_1_stride0, in_1_stride1, in_1_stride2, in_1_stride3,
    in_2_stride0, in_2_stride1, in_2_stride2, in_2_stride3,
    out0_stride0, out0_stride1, out0_stride2,
    out1_stride0, out1_stride1, out1_stride2,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    b = pid_m // M
    i = pid_m % M

    k_start = pid_k * BLOCK_K
    k_offsets = k_start + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < N

    # Read in_2[b, i, 0, k] - the input vector
    in_2_base = b * in_2_stride0 + i * in_2_stride1
    in_2_offsets = in_2_base + k_offsets * in_2_stride3
    in_2_vals = tl.load(in_2_ptr + in_2_offsets, mask=k_mask, other=0.0)

    # Read in_1[0, 0, 0, k] - scale for unbind[0]
    in_1_0_offsets = k_offsets * in_1_stride3
    in_1_0_vals = tl.load(in_1_ptr + in_1_0_offsets, mask=k_mask, other=0.0)

    # Read in_1[0, 0, 1, k] - scale for unbind[1]
    in_1_1_offsets = in_1_stride2 + k_offsets * in_1_stride3
    in_1_1_vals = tl.load(in_1_ptr + in_1_1_offsets, mask=k_mask, other=0.0)

    # Read in_0[0, k] - bias for unbind[0]
    in_0_0_offsets = k_offsets * in_0_stride1
    in_0_0_vals = tl.load(in_0_ptr + in_0_0_offsets, mask=k_mask, other=0.0)

    # Read in_0[1, k] - bias for unbind[1]
    in_0_1_offsets = in_0_stride0 + k_offsets * in_0_stride1
    in_0_1_vals = tl.load(in_0_ptr + in_0_1_offsets, mask=k_mask, other=0.0)

    # Compute fused multiply + add for both unbind slices
    # unbind[0][b, i, k] = in_2[b,i,0,k] * in_1[0,0,0,k] + in_0[0,k]
    val0 = in_2_vals * in_1_0_vals + in_0_0_vals
    # unbind[1][b, i, k] = in_2[b,i,0,k] * in_1[0,0,1,k] + in_0[1,k]
    val1 = in_2_vals * in_1_1_vals + in_0_1_vals

    # Write out0[b, i, k] = unbind[0] (tmp_4)
    out0_base = b * out0_stride0 + i * out0_stride1
    out0_offsets = out0_base + k_offsets * out0_stride2
    tl.store(out0_ptr + out0_offsets, val0, mask=k_mask)

    # Write out1[b, k, i] = unbind[1] permuted (tmp_6)
    out1_offsets = b * out1_stride0 + k_offsets * out1_stride1 + i * out1_stride2
    tl.store(out1_ptr + out1_offsets, val1, mask=k_mask)


@torch.fx.wrap
def fused_mul_add_unbind_permute(in_0, in_1, in_2):
    B = in_2.shape[0]
    M = in_2.shape[1]  # 17
    N = in_2.shape[3]  # 128

    # out0: unbind[0] shape [B, 17, 128] (tmp_4)
    out0 = torch.empty((B, M, N), dtype=in_2.dtype, device=in_2.device)
    # out1: unbind[1].permute(0,2,1) shape [B, 128, 17] (tmp_6)
    out1 = torch.empty((B, N, M), dtype=in_2.dtype, device=in_2.device)

    grid = lambda meta: (B * M, triton.cdiv(N, meta['BLOCK_K']))

    fused_mul_add_unbind_permute_kernel[grid](
        in_0, in_1, in_2, out0, out1,
        B, M, N,
        in_0.stride(0), in_0.stride(1),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        out0.stride(0), out0.stride(1), out0.stride(2),
        out1.stride(0), out1.stride(1), out1.stride(2),
    )

    # Return order matches model: (tmp_6, tmp_4) = (out1, out0)
    return (out1, out0)


def replacement_func():
    return fused_mul_add_unbind_permute