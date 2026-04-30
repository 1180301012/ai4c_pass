import torch
import triton
import triton.language as tl

# Pattern matching function for: matmul -> div -> softmax -> dropout -> matmul -> permute -> contiguous -> view
def pattern(in_0, in_1, in_2):
    tmp_0 = torch.matmul(in_0, in_1)
    tmp_1 = tmp_0 / 5.656854249492381
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    tmp_4 = torch.matmul(tmp_3, in_2)
    tmp_5 = tmp_4.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view((1, 16384, 32))
    return tmp_7

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_q0, stride_q1, stride_q2, stride_q3,
    stride_k0, stride_k1, stride_k2, stride_k3,
    stride_v0, stride_v1, stride_v2, stride_v3,
    stride_out0, stride_out1, stride_out2, stride_out3,
    scale,
    N, D, K,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid: processes (b*h*n) positions
    pid = tl.program_id(0)
    
    # Calculate which batch, head, and position this program handles
    # Q shape: [B, H, N, D] - we process position n for each (b, h)
    b = pid // (N * 1)  # Since H=1 in the first pattern
    remainder = pid % (N * 1)
    h = remainder // N
    n = remainder % N
    
    # Compute row offset for Q
    q_row_offset = b * stride_q0 + h * stride_q1 + n * stride_q2
    
    # Load Q row: [D]
    offs_d = tl.arange(0, D)
    q_offsets = q_row_offset + offs_d * stride_q3
    q_mask = offs_d < D
    q = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
    
    # Compute Q @ K^T for this position
    # K shape: [B, H, D, K], result is [K]
    scores = tl.zeros((K,), dtype=tl.float32)
    
    for d in range(D):
        k_base = b * stride_k0 + h * stride_k1 + d * stride_k2
        k_offsets = k_base + offs_d * stride_k3
        k_mask = offs_d < D
        k = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)
        scores += q[d] * k
    
    # Scale
    scores = scores * scale
    
    # Softmax over K
    max_score = tl.max(scores, axis=0)
    exp_scores = tl.exp(scores - max_score)
    sum_exp = tl.sum(exp_scores, axis=0)
    attn = exp_scores / sum_exp
    
    # Compute attn @ V
    # V shape: [B, H, K, D]
    output = tl.zeros((D,), dtype=tl.float32)
    
    for k in range(K):
        v_base = b * stride_v0 + h * stride_v1 + k * stride_v2
        v_offsets = v_base + offs_d * stride_v3
        v_mask = offs_d < D
        v = tl.load(v_ptr + v_offsets, mask=v_mask, other=0.0)
        output += attn[k] * v
    
    # Store output: [B, H, N, D]
    out_base = b * stride_out0 + h * stride_out1 + n * stride_out2
    out_offsets = out_base + offs_d * stride_out3
    tl.store(out_ptr + out_offsets, output, mask=q_mask)


@torch.fx.wrap
def fused_attention_kernel_wrapper(q, k, v):
    """
    Fused scaled dot-product attention kernel.
    Q: [B, H, N, D]
    K: [B, H, D, K] 
    V: [B, H, K, D]
    Output: [1, N, D] after reshape
    """
    scale = 1.0 / 5.656854249492381
    
    B, H, N, D = q.shape
    K = k.shape[3]
    
    # Allocate output in [B, H, N, D] format
    out = torch.empty((B, H, N, D), dtype=q.dtype, device=q.device)
    
    # Launch one program per (b, h, n) position
    num_programs = B * H * N
    
    fused_attention_kernel[(num_programs,)](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale, N, D, K,
    )
    
    # Reshape to match pattern output: [1, N, D]
    out = out.permute(0, 2, 1, 3)
    out = out.contiguous()
    out = out.view(1, N, D)
    
    return out


def replacement_func():
    return fused_attention_kernel_wrapper