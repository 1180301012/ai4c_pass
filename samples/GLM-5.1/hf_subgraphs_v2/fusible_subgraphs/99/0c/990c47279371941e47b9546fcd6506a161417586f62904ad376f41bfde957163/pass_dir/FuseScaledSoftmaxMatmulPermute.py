import torch
import triton
import triton.language as tl


def pattern(in_0: torch.Tensor, in_1: torch.Tensor):
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    matmul = torch.matmul(tmp_1, in_1)
    tmp_3 = matmul.permute(0, 2, 1)
    return (tmp_3,)


def replacement_args(in_0: torch.Tensor, in_1: torch.Tensor):
    return (in_0, in_1)


@triton.jit
def fused_scaled_softmax_matmul_permute_kernel(
    # Pointers
    q_ptr,  # in_0: [B, M, K] - query/similarity map
    v_ptr,  # in_1: [B, K, N] - value
    out_ptr,  # output: [B, N, M] - transposed result
    # Dimensions
    B, M, K, N,
    # Strides for q: [B, M, K]
    q_stride_b, q_stride_m, q_stride_k,
    # Strides for v: [B, K, N]
    v_stride_b, v_stride_k, v_stride_n,
    # Strides for out: [B, N, M]
    out_stride_b, out_stride_n, out_stride_m,
    # Scale factor
    scale: tl.constexpr,
    # Block sizes
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one row: (batch_b, query_m)
    # Computes softmax(q[b, m, :]) then dot product with v[b, :, :] for a block of N
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Compute softmax of q[b, m, :] with scale factor
    # q offsets: q[b, m, k] for k in [0, K)
    q_k_offsets = tl.arange(0, BLOCK_K)
    q_k_mask = q_k_offsets < K
    q_base = q_ptr + pid_b * q_stride_b + pid_m * q_stride_m
    q_row = tl.load(q_base + q_k_offsets * q_stride_k, mask=q_k_mask, other=0.0)
    
    # Scale and compute softmax in fp32 for numerical stability
    q_scaled = q_row.to(tl.float32) * scale
    q_max = tl.max(q_scaled, axis=0)
    q_exp = tl.exp(q_scaled - q_max)
    q_sum = tl.sum(q_exp, axis=0)
    softmax_row = q_exp / q_sum  # [K] in fp32

    # Compute output for a block of N columns
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Load v[b, :, n_block] - we need all K rows for each n in the block
    # v[b, k, n] for k in [0, K), n in n_offsets
    # We load [BLOCK_K, BLOCK_N] block of v
    v_base = v_ptr + pid_b * v_stride_b
    v_block = tl.load(
        v_base + q_k_offsets[:, None] * v_stride_k + n_offsets[None, :] * v_stride_n,
        mask=q_k_mask[:, None] & n_mask[None, :],
        other=0.0,
    )  # [BLOCK_K, BLOCK_N]

    # Dot product: softmax_row[k] * v[b, k, n] summed over k
    # softmax_row is [K], v_block is [K, BLOCK_N]
    dot = tl.sum(softmax_row[:, None] * v_block.to(tl.float32), axis=0)  # [BLOCK_N]

    # Write to output[b, n_block, m] - output is [B, N, M]
    out_base = out_ptr + pid_b * out_stride_b + pid_m * out_stride_m
    tl.store(out_base + n_offsets * out_stride_n, dot.to(out_ptr.dtype.element_ty), mask=n_mask)


@torch.fx.wrap
def fused_scaled_softmax_matmul_permute(in_0: torch.Tensor, in_1: torch.Tensor):
    B, M, K = in_0.shape
    _, _, N = in_1.shape
    
    # Output shape is [B, N, M] (permuted from [B, M, N])
    out = torch.empty((B, N, M), dtype=in_0.dtype, device=in_0.device)
    
    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_N = 64
    
    grid = (B, M, triton.cdiv(N, BLOCK_N))
    
    fused_scaled_softmax_matmul_permute_kernel[grid](
        q_ptr=in_0,
        v_ptr=in_1,
        out_ptr=out,
        B=B, M=M, K=K, N=N,
        q_stride_b=in_0.stride(0), q_stride_m=in_0.stride(1), q_stride_k=in_0.stride(2),
        v_stride_b=in_1.stride(0), v_stride_k=in_1.stride(1), v_stride_n=in_1.stride(2),
        out_stride_b=out.stride(0), out_stride_n=out.stride(1), out_stride_m=out.stride(2),
        scale=0.0625,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
    )
    
    return (out,)


def replacement_func():
    return fused_scaled_softmax_matmul_permute