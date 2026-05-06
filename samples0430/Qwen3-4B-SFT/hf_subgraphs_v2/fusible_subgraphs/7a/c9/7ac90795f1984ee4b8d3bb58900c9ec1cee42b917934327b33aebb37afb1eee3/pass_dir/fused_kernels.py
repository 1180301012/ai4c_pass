"""
Shared Triton kernels for matmul+reshape and transpose operations.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel: fused matmul + reshape  (in_1: [B,H,K], in_0: [B,K,1] → [BH,H_dim])
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 1},  num_warps=2),
        triton.Config({'BLOCK_H': 2},  num_warps=2),
        triton.Config({'BLOCK_H': 4},  num_warps=2),
        triton.Config({'BLOCK_H': 8},  num_warps=4),
        triton.Config({'BLOCK_H': 16}, num_warps=4),
        triton.Config({'BLOCK_H': 32}, num_warps=4),
        triton.Config({'BLOCK_H': 64}, num_warps=8),
        triton.Config({'BLOCK_H': 128}, num_warps=8),
    ],
    key=['B', 'H_DIM_IN1', 'H_DIM_IN2'],
)
@triton.jit
def matmul_reshape_kernel(
    in1_ptr, in0_ptr, out_ptr,
    B, H_DIM_IN1, H_DIM_IN2, K_DIM,
    stride_in1_b, stride_in1_h, stride_in0_b,
    BLOCK_H: tl.constexpr,
):
    """
    2D grid: dim0 = batch b, dim1 = h-block.
    Computes: out[b*H_DIM_IN2 + h] = sum_k(in1[b,h,k] * in0[b,k])
    """
    b       = tl.program_id(0)
    h_block = tl.program_id(1)

    h_start   = h_block * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h    = h_offsets < H_DIM_IN1

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for k_i in range(K_DIM):
        in1_vals = tl.load(
            in1_ptr + b * stride_in1_b + h_offsets * stride_in1_h + k_i,
            mask=mask_h, other=0.0
        ).to(tl.float32)
        in0_vals = tl.load(
            in0_ptr + b * stride_in0_b + k_i,
            mask=mask_h, other=0.0
        ).to(tl.float32)
        acc += in1_vals * in0_vals

    # Output is [B*H_DIM_IN1, H_DIM_IN2] contiguous; row b starts at b*H_DIM_IN2
    out_offsets = b * H_DIM_IN2 + h_offsets
    tl.store(out_ptr + out_offsets, acc.to(out_ptr.dtype.element_ty), mask=mask_h)


# ---------------------------------------------------------------------------
# Kernel: transpose last two dims  [..., D_N2, D_N1] → [..., D_N1, D_N2]
# flat_input = row * D_N1 + col   <==>   row = flat // D_N1,  col = flat % D_N1
# flat_output = col * (N // D_N1) + row
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 256}, num_warps=4),
        triton.Config({'BLOCK': 512}, num_warps=4),
        triton.Config({'BLOCK': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def transpose_last2_kernel(
    in_ptr, out_ptr,
    N, D_N1,
    BLOCK: tl.constexpr,
):
    pid         = tl.program_id(0)
    block_start = pid * BLOCK
    offsets     = block_start + tl.arange(0, BLOCK)
    mask        = offsets < N

    row = offsets // D_N1
    col = offsets % D_N1
    out_offsets = col * (N // D_N1) + row

    vals = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + out_offsets, vals, mask=mask)


@torch.fx.wrap
def fuse_matmul_reshape_16(in_0, in_1):
    """
    Fused matmul + reshape for H_DIM_IN2=16.
    in_0: [B, K, 1], in_1: [B, H, K]  →  out: [B*H, 16]
    """
    B_dim = in_1.shape[0]
    K_dim = in_0.shape[1]
    H_dim1 = in_1.shape[1]
    H_dim2 = 16

    out = torch.empty((B_dim * H_dim1, H_dim2), dtype=in_0.dtype, device=in_0.device)

    stride_in1_b = in_1.stride(0)
    stride_in1_h = in_1.stride(1)
    stride_in0_b = in_0.stride(0)

    grid = lambda meta: (B_dim, triton.cdiv(H_dim1, meta['BLOCK_H']))

    matmul_reshape_kernel[grid](
        in_1, in_0, out,
        B_dim, H_dim1, H_dim2, K_dim,
        stride_in1_b, stride_in1_h, stride_in0_b,
    )
    return out


@torch.fx.wrap
def fuse_matmul_reshape_128(in_0, in_1):
    """
    Fused matmul + reshape for H_DIM_IN2=128.
    in_0: [B, K, 1], in_1: [B, H, K]  →  out: [B*H, 128]
    """
    B_dim = in_1.shape[0]
    K_dim = in_0.shape[1]
    H_dim1 = in_1.shape[1]
    H_dim2 = 128

    out = torch.empty((B_dim * H_dim1, H_dim2), dtype=in_0.dtype, device=in_0.device)

    stride_in1_b = in_1.stride(0)
    stride_in1_h = in_1.stride(1)
    stride_in0_b = in_0.stride(0)
    out_row_stride = H_dim2

    BH      = B_dim * H_dim1
    grid    = lambda meta: (BH,)

    matmul_reshape_kernel[grid](
        in_1, in_0, out,
        B_dim, H_dim1, H_dim2, K_dim,
        stride_in1_b, stride_in1_h, stride_in0_b,
        out_row_stride,
    )
    return out


@torch.fx.wrap
def fuse_matmul_reshape_384(in_0, in_1):
    """
    Fused matmul + reshape for H_DIM_IN2=384.
    in_0: [B, K, 1], in_1: [B, H, K]  →  out: [B*H, 384]
    """
    B_dim = in_1.shape[0]
    K_dim = in_0.shape[1]
    H_dim1 = in_1.shape[1]
    H_dim2 = 384

    out = torch.empty((B_dim * H_dim1, H_dim2), dtype=in_0.dtype, device=in_0.device)

    stride_in1_b = in_1.stride(0)
    stride_in1_h = in_1.stride(1)
    stride_in0_b = in_0.stride(0)

    grid = lambda meta: (B_dim, triton.cdiv(H_dim1, meta['BLOCK_H']))

    matmul_reshape_kernel[grid](
        in_1, in_0, out,
        B_dim, H_dim1, H_dim2, K_dim,
        stride_in1_b, stride_in1_h, stride_in0_b,
    )
    return out


@torch.fx.wrap
def triton_transpose_last2(in_2):
    """
    Transpose the last two dimensions of in_2.
    in_2: [..., D_N2, D_N1]  →  [..., D_N1, D_N2]
    """
    s       = in_2.shape
    N       = in_2.numel()
    D_N1    = s[-1]
    D_N2    = N // D_N1
    new_s   = list(s[:-2]) + [D_N1, D_N2]
    out     = torch.empty(new_s, dtype=in_2.dtype, device=in_2.device)
    grid    = lambda meta: ((N + meta['BLOCK'] - 1) // meta['BLOCK'],)
    transpose_last2_kernel[grid](in_2, out, N, D_N1)
    return out