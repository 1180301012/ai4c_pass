import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    einsum = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    in_3 += einsum
    tmp_3 = in_3 * in_0
    tmp_4 = tmp_3 + in_2
    tmp_5 = tmp_4.contiguous()
    return (tmp_5,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def fused_einsum_mul_add_kernel(
    in_4_ptr, in_1_ptr, in_3_ptr, in_0_ptr, in_2_ptr, out_ptr,
    B, C, H, W, J,
    in_4_stride_b, in_4_stride_c, in_4_stride_h, in_4_stride_j,
    in_1_stride_b, in_1_stride_h, in_1_stride_w, in_1_stride_j,
    in_3_stride_b, in_3_stride_c, in_3_stride_h, in_3_stride_w,
    in_2_stride_b, in_2_stride_c, in_2_stride_h, in_2_stride_w,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs - 3D grid over (bh_batch, m_tiles, n_tiles)
    bh_idx = tl.program_id(0)
    m_tile = tl.program_id(1)
    n_tile = tl.program_id(2)

    # Map batch index to (b, h) indices
    b_idx = bh_idx // H
    h_idx = bh_idx % H

    # Output tile offsets
    c_start = m_tile * BLOCK_M
    w_start = n_tile * BLOCK_N
    c_offsets = c_start + tl.arange(0, BLOCK_M)
    w_offsets = w_start + tl.arange(0, BLOCK_N)
    c_mask = c_offsets < C
    w_mask = w_offsets < W

    # Accumulate the einsum result over J dimension
    # einsum('bchj,bhwj->bchw', in_4, in_1):
    #   output[b,c,h,w] = sum_j in_4[b,c,h,j] * in_1[b,h,w,j]
    # View as batched matmul: for each (b,h), A[C,J] @ B^T[W,J] = [C,W]
    # A = in_4[b,:,h,:] shape [C, J]
    # B^T = transpose of in_1[b,h,:,:] shape [J, W]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, J, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < J

        # Load A tile: in_4[b, c_range, h, k_range] -> [BLOCK_M, BLOCK_K]
        a_ptrs = in_4_ptr + b_idx * in_4_stride_b + h_idx * in_4_stride_h
        a_ptrs = a_ptrs + c_offsets[:, None] * in_4_stride_c + k_offsets[None, :] * in_4_stride_j
        a_mask = c_mask[:, None] & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B^T tile: in_1[b, h, k_range, w_range] -> [BLOCK_K, BLOCK_N]
        # We access in_1[b,h,w,j] but arranged as [j,w] for matmul right side
        b_ptrs = in_1_ptr + b_idx * in_1_stride_b + h_idx * in_1_stride_h
        b_ptrs = b_ptrs + k_offsets[:, None] * in_1_stride_j + w_offsets[None, :] * in_1_stride_w
        b_mask = k_mask[:, None] & w_mask[None, :]
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # tl.dot: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        # For fp16/bf16 inputs, tl.dot uses tensor cores and accumulates in fp32
        acc += tl.dot(a, b_vals, allow_tf32=True)

    # Fused elementwise: result = (in_3 + einsum) * in_0 + in_2
    # Load in_3[b, c_range, h, w_range] -> [BLOCK_M, BLOCK_N]
    in_3_ptrs = in_3_ptr + b_idx * in_3_stride_b + h_idx * in_3_stride_h
    in_3_ptrs = in_3_ptrs + c_offsets[:, None] * in_3_stride_c + w_offsets[None, :] * in_3_stride_w
    in_3_mask = c_mask[:, None] & w_mask[None, :]
    in_3_vals = tl.load(in_3_ptrs, mask=in_3_mask, other=0.0).to(tl.float32)

    # Load in_0 (scalar)
    in_0_val = tl.load(in_0_ptr).to(tl.float32)

    # Load in_2[b, c_range, h, w_range] -> [BLOCK_M, BLOCK_N]
    in_2_ptrs = in_2_ptr + b_idx * in_2_stride_b + h_idx * in_2_stride_h
    in_2_ptrs = in_2_ptrs + c_offsets[:, None] * in_2_stride_c + w_offsets[None, :] * in_2_stride_w
    in_2_mask = c_mask[:, None] & w_mask[None, :]
    in_2_vals = tl.load(in_2_ptrs, mask=in_2_mask, other=0.0).to(tl.float32)

    # Fused computation: (in_3 + einsum) * in_0 + in_2
    result = (in_3_vals + acc) * in_0_val + in_2_vals

    # Store output[b, c_range, h, w_range]
    out_ptrs = out_ptr + b_idx * out_stride_b + h_idx * out_stride_h
    out_ptrs = out_ptrs + c_offsets[:, None] * out_stride_c + w_offsets[None, :] * out_stride_w
    out_mask = c_mask[:, None] & w_mask[None, :]
    tl.store(out_ptrs, result, mask=out_mask)


@torch.fx.wrap
def fused_einsum_mul_add(in_0, in_1, in_2, in_3, in_4):
    # Determine dimensions from input shapes
    # in_4: [B, C, H, J] e.g. [B, 512, 64, 64]
    # in_1: [B, H, W, J] e.g. [B, 64, 64, 64]
    # in_3: [B, C, H, W] e.g. [B, 512, 64, 64]
    # in_2: [B, C, H, W] e.g. [B, 512, 64, 64]
    # in_0: scalar
    B = in_4.shape[0]
    C = in_4.shape[1]
    H = in_4.shape[2]
    J = in_4.shape[3]
    W = in_1.shape[2]  # in_1 is [B, H, W, J]

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 64

    # Allocate contiguous output
    out = torch.empty(B, C, H, W, dtype=in_4.dtype, device=in_4.device)

    # Grid: (B*H batches, M_tiles, N_tiles)
    batch_size = B * H
    M_tiles = (C + BLOCK_M - 1) // BLOCK_M
    N_tiles = (W + BLOCK_N - 1) // BLOCK_N

    grid = (batch_size, M_tiles, N_tiles)

    fused_einsum_mul_add_kernel[grid](
        in_4, in_1, in_3, in_0, in_2, out,
        B, C, H, W, J,
        in_4.stride(0), in_4.stride(1), in_4.stride(2), in_4.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return (out,)


def replacement_func():
    return fused_einsum_mul_add