import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['H', 'S_q', 'S_k', 'D'],
)
@triton.jit
def matmul_permute_kernel(
    weights_ptr,  # [B, H, S_q, S_k]  attention weights (softmax output)
    values_ptr,   # [B, H, S_k, D]    value tensor
    out_ptr,      # [B, S_q, H, D]    output in permuted layout
    H, S_q, S_k, D,
    BLOCK_D: tl.constexpr,   # = next_power_of_2(D)
    IS_BF16: tl.constexpr,
):
    """
    Replaces: matmul(weights, values).permute(0,2,1,3).contiguous()

    Key design: one program per (b, h, q).  The inner loop over k loads
    ONE scalar weight + ONE row of D values, so register usage is only
    ~2*BLOCK_D + 1 float32 (≤129 for D=64).  Values[b,h,:,:] fit in L1
    cache and are reused across the S_q programs that share the same (b,h),
    giving the same effective DRAM traffic as cuBLAS while writing the
    output directly in permuted [B, S_q, H, D] order.
    """
    pid   = tl.program_id(0)
    h_idx = pid % H
    q_idx = (pid // H) % S_q
    b_idx = pid // (H * S_q)

    w_base = b_idx * H * S_q * S_k + h_idx * S_q * S_k + q_idx * S_k
    v_base = b_idx * H * S_k * D   + h_idx * S_k * D

    d_range = tl.arange(0, BLOCK_D)
    d_mask  = d_range < D

    # Accumulate: output[d] = sum_k  weights[k] * values[k, d]
    output = tl.zeros([BLOCK_D], dtype=tl.float32)
    for k in range(S_k):
        w = tl.load(weights_ptr + w_base + k).to(tl.float32)
        v = tl.load(values_ptr  + v_base  + k * D + d_range,
                    mask=d_mask, other=0.0).to(tl.float32)
        output = output + w * v

    # Store in permuted layout: out[b, q, h, d]
    o_base = b_idx * S_q * H * D + q_idx * H * D + h_idx * D
    if IS_BF16:
        tl.store(out_ptr + o_base + d_range, output.to(tl.bfloat16), mask=d_mask)
    else:
        tl.store(out_ptr + o_base + d_range, output.to(tl.float16),  mask=d_mask)


@torch.fx.wrap
def fused_matmul_permute(x, y):
    """
    Fused replacement for: torch.matmul(x, y).permute(0,2,1,3).contiguous()
    x: [B, H, S_q, S_k]  (attention weights after softmax/dropout)
    y: [B, H, S_k, D]    (value tensor)
    Returns: [B, S_q, H, D]
    """
    B, H, S_q, S_k = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    D = y.shape[3]

    out     = torch.empty((B, S_q, H, D), dtype=x.dtype, device=x.device)
    BLOCK_D = triton.next_power_of_2(D)
    IS_BF16 = (x.dtype == torch.bfloat16)

    matmul_permute_kernel[(B * H * S_q,)](
        x, y, out,
        H, S_q, S_k, D,
        BLOCK_D=BLOCK_D,
        IS_BF16=IS_BF16,
    )
    return out