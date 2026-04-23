import torch
import triton
import triton.language as tl

# ===========================================================================
# Shared Fused Attention Kernel - Core computation
# Handles both 256x256 (16 heads) and 64x64 (8 heads) patterns
# ===========================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_D': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_D': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_D': 64}, num_stages=3, num_warps=4),
    ],
    key=['N', 'D'],
)
@triton.jit
def fused_attention_kernel_impl(
    scores_ptr, value_ptr, output_ptr,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    stride_scores_b, stride_scores_h, stride_scores_n1, stride_scores_n2,
    stride_value_b, stride_value_h, stride_value_n, stride_value_d,
    stride_out_b, stride_out_h, stride_out_d, stride_out_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (B*H*num_blocks_m, num_blocks_n)
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(N, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    
    # Decode program id
    pid_m = pid // num_blocks_n
    pid_n = pid % num_blocks_n
    
    # Offsets for this block
    off_bh = pid_m // num_blocks_m * H + (pid_m % num_blocks_m)
    off_h = pid_m % num_blocks_m
    
    # Offsets for scores [B, H, N, N]
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Load scores block
    scores_offsets = (
        (m_offsets[:, None] * stride_scores_n1) + 
        (n_offsets[None, :] * stride_scores_n2)
    )
    scores_mask = (m_offsets[:, None] < N) & (n_offsets[None, :] < N)
    scores_block = tl.load(scores_ptr + scores_offsets, mask=scores_mask, other=float('-inf'))
    
    # Softmax computation
    scores_max = tl.max(scores_block, axis=1, keepdim=True)
    scores_exp = tl.exp(scores_block - scores_max)
    scores_sum = tl.sum(scores_exp, axis=1, keepdim=True) + 1e-8
    attn_weights = scores_exp / scores_sum
    
    # Load values [B, H, N, D] for the column block
    v_offsets = (
        (n_offsets[:, None] * stride_value_n) + 
        (tl.arange(0, BLOCK_D)[None, :] * stride_value_d)
    )
    v_mask = (n_offsets[:, None] < N) & (tl.arange(0, BLOCK_D)[None, :] < D)
    v_block = tl.load(value_ptr + v_offsets, mask=v_mask, other=0.0)
    
    # Compute output block [M, D]
    out_block = tl.dot(attn_weights.to(v_block.dtype), v_block)
    
    # Store output [B, H, D, N]
    out_m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_d_offsets = tl.arange(0, BLOCK_D)
    out_offsets = (
        (out_m_offsets[:, None] * stride_out_d) +
        (out_d_offsets[None, :] * stride_out_n)
    )
    out_mask = (out_m_offsets[:, None] < D) & (out_d_offsets[None, :] < N)
    tl.store(output_ptr + out_offsets, out_block, mask=out_mask)


@torch.fx.wrap
def fused_attention_dispatcher(q, k, v, rel_bias, scale_in, route="attention_256"):
    """
    Fused attention dispatcher: routes to appropriate kernel based on route string.
    All pattern passes share this single replacement function.
    """
    if route == "attention_256":
        # 256x256 pattern: B=4, H=16, N=256, D=128
        B, H, N, _ = q.shape
        _, _, _, D = v.shape
        
        output = torch.empty(B, H, D, N, device=q.device, dtype=q.dtype)
        
        grid = (B * H, 1)
        fused_attention_kernel_impl[grid](
            q, v, output,
            B, H, N, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        )
        return output
    elif route == "attention_64":
        # 64x64 pattern: B=4, H=8, N=64, D=128
        B, H, N, _ = q.shape
        _, _, _, D = v.shape
        
        output = torch.empty(B, H, D, N, device=q.device, dtype=q.dtype)
        
        grid = (B * H, 1)
        fused_attention_kernel_impl[grid](
            q, v, output,
            B, H, N, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        )
        return output
    else:
        # Fallback: should not reach here
        return q @ v