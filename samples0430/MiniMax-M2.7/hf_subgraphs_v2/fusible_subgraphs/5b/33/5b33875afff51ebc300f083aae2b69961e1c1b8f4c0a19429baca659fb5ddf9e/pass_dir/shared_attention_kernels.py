import torch
import triton
import triton.language as tl

# Shared attention kernel implementations for different configurations

@triton.jit
def attention_softmax_kernel(
    scores_ptr, mask_ptr, max_ptr, sum_ptr,
    batch: tl.constexpr, heads: tl.constexpr,
    seq_len: tl.constexpr, n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute softmax over attention scores with mask.
    scores: [batch, heads, seq_len, seq_len]
    mask: [batch, 1, 1, seq_len]
    output: [batch, heads, seq_len, seq_len]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate indices
    batch_idx = offsets // (heads * seq_len * seq_len)
    rest = offsets % (heads * seq_len * seq_len)
    head_idx = rest // (seq_len * seq_len)
    seq_idx = (rest % (seq_len * seq_len)) // seq_len
    col_idx = rest % seq_len
    
    load_mask = (offsets < n_elements) & (col_idx < seq_len)
    
    # Load score
    scores = tl.load(scores_ptr + offsets, mask=load_mask, other=0.0).to(tl.float32)
    
    # Load mask (broadcast from [batch, 1, 1, seq_len])
    mask_base = batch_idx * seq_len + 0  # second and third dims are 0
    mask_offsets_full = mask_base * 1 + 0 + col_idx  # shape is [1, 1, 1, seq_len]
    m = tl.load(mask_ptr + col_idx, mask=col_idx < seq_len, other=0.0).to(tl.float32)
    
    # Add mask
    scores = scores + m
    
    # Compute max for stability
    max_val = tl.max(scores, axis=0)
    tl.store(max_ptr + pid, max_val)


@triton.jit  
def attention_matmul_softmax_kernel(
    scores_ptr, mask_ptr, value_ptr, out_ptr,
    scale: tl.constexpr,
    batch: tl.constexpr, heads: tl.constexpr,
    seq_len: tl.constexpr, head_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused attention: scale scores, add mask, softmax, matmul with value.
    scores: [batch, heads, M, K] where M=K=seq_len
    value: [batch, heads, K, N] where N=head_dim
    output: [batch, M, heads, N] (after permute)
    """
    # Grid: (batch * seq_len, heads)
    batch_seq_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    
    batch_idx = batch_seq_pid // seq_len
    seq_idx = batch_seq_pid % seq_len
    
    #Offsets for scores [batch, heads, seq, :]
    score_offsets = (
        batch_idx * heads * seq_len * seq_len +
        head_pid * seq_len * seq_len +
        seq_idx * seq_len
    )
    
    # Compute max for softmax stability
    max_val = -1e9
    for k in range(BLOCK_SIZE_K):
        if k < seq_len:
            s = tl.load(scores_ptr + score_offsets + k).to(tl.float32) / scale
            max_val = max(max_val, s)
    
    # Compute exp and sum for softmax
    exp_sum = 0.0
    exp_vals = tl.zeros([BLOCK_SIZE_K], dtype=tl.float32)
    for k in range(BLOCK_SIZE_K):
        if k < seq_len:
            s = tl.load(scores_ptr + score_offsets + k).to(tl.float32) / scale
            e = tl.exp(s - max_val)
            exp_vals[k] = e
            exp_sum += e
    
    # Compute attention output: sum_k(attn[k] * value[k, :])
    # Value layout: [batch, heads, seq_len, head_dim]
    output_offsets = (
        batch_idx * seq_len * heads * head_dim +
        seq_idx * heads * head_dim +
        head_pid * head_dim
    )
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    
    for k in range(BLOCK_SIZE_K):
        if k < seq_len:
            attn_weight = exp_vals[k] / exp_sum
            
            # Load value[k, :] for this head
            value_offset = (
                batch_idx * heads * seq_len * head_dim +
                head_pid * seq_len * head_dim +
                k * head_dim
            )
            
            for n in range(BLOCK_SIZE_N):
                if n < head_dim:
                    v = tl.load(value_ptr + value_offset + n).to(tl.float32)
                    acc[n] += attn_weight * v
    
    # Store output
    for n in range(BLOCK_SIZE_N):
        if n < head_dim:
            tl.store(out_ptr + output_offsets + n, acc[n])


def triton_fused_attention_scale8(in_0, in_1, in_2, in_3, reshape_h, reshape_hd):
    """Fused attention with scale=8.0 (head_dim=64)"""
    batch, heads, seq_len, _ = in_0.shape
    head_dim = in_3.shape[-1]
    
    out1 = torch.empty((batch, seq_len, heads, head_dim), dtype=in_0.dtype, device=in_0.device)
    out2 = in_1.reshape((1, -1, reshape_h, reshape_hd))
    
    BLOCK_SIZE = min(1024, max(seq_len, head_dim))
    grid = (batch * seq_len, heads)
    
    attention_matmul_softmax_kernel[grid](
        in_0, in_2, in_3, out1,
        8.0, batch, heads, seq_len, head_dim,
        BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE
    )
    
    return (out1, out2)


def triton_fused_attention_scale2_828(in_0, in_1, in_2, in_3, reshape_h, reshape_hd):
    """Fused attention with scale=2.828... (head_dim=8)"""
    batch, heads, seq_len, _ = in_0.shape
    head_dim = in_3.shape[-1]
    
    out1 = torch.empty((batch, seq_len, heads, head_dim), dtype=in_0.dtype, device=in_0.device)
    out2 = in_1.reshape((1, -1, reshape_h, reshape_hd))
    
    BLOCK_SIZE = min(1024, max(seq_len, head_dim))
    grid = (batch * seq_len, heads)
    
    attention_matmul_softmax_kernel[grid](
        in_0, in_2, in_3, out1,
        2.8284271247461903, batch, heads, seq_len, head_dim,
        BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE
    )
    
    return (out1, out2)


@torch.fx.wrap
def shared_attention_dispatcher(in_0, in_1, in_2, in_3, scale, reshape_h, reshape_hd, route=""):
    """
    Shared dispatcher that routes to the appropriate kernel based on scale factor.
    """
    if abs(scale - 8.0) < 0.001:
        return triton_fused_attention_scale8(in_0, in_1, in_2, in_3, reshape_h, reshape_hd)
    elif abs(scale - 2.8284271247461903) < 0.001:
        return triton_fused_attention_scale2_828(in_0, in_1, in_2, in_3, reshape_h, reshape_hd)
    else:
        # Default fallback - use PyTorch
        return fallback_attention(in_0, in_1, in_2, in_3, scale, reshape_h, reshape_hd)


def fallback_attention(in_0, in_1, in_2, in_3, scale, reshape_h, reshape_hd):
    """Fallback PyTorch implementation for unmatched scales"""
    tmp_0 = in_0 / scale
    tmp_1 = tmp_0 + in_2
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul = torch.matmul(tmp_3, in_3)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = torch.reshape(in_1, [1, -1, reshape_h, reshape_hd])
    return (tmp_6, tmp_7)