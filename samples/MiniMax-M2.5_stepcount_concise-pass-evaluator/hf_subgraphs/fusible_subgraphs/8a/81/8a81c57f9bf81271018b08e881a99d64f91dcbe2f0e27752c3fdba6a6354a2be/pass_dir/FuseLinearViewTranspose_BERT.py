import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 64}, num_stages=3, num_warps=1),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 128}, num_stages=3, num_warps=1),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 256}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 512}, num_stages=3, num_warps=4),
    ],
    key=['head_dim'],
)
@triton.jit
def fused_linear_view_transpose_bert_kernel(
    hidden_states_ptr, weight_ptr, bias_ptr,
    output_ptr,
    batch_size, seq_len, hidden_dim,
    num_heads, head_dim,
    stride_hs, stride_s, stride_h,
    stride_w, stride_v,
    stride_o, stride_b, stride_s_out, stride_h_out,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Fused kernel for BERT (2-head, 128 hidden dim):
    1. Linear transformation (matmul + bias)
    2. View/Reshape to (batch, seq, num_heads, head_dim)
    3. Transpose to (batch, num_heads, seq, head_dim)
    """
    batch_idx = tl.program_id(0)
    hs_offset = batch_idx * stride_hs
    output_offset = batch_idx * stride_o
    
    for seq_idx in range(seq_len):
        for head_idx in range(num_heads):
            for dim_idx in range(head_dim):
                col_idx = head_idx * head_dim + dim_idx
                dot_prod = tl.zeros((1,), dtype=tl.float32)
                
                for k in range(0, hidden_dim, BLOCK_N):
                    weight_mask = k + tl.arange(0, BLOCK_N) < hidden_dim
                    weight_offsets = (k + tl.arange(0, BLOCK_N)) * stride_w + col_idx
                    weight_vals = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)
                    
                    hs_mask = k + tl.arange(0, BLOCK_N) < hidden_dim
                    hs_offsets = hs_offset + seq_idx * stride_s + (k + tl.arange(0, BLOCK_N))
                    hs_vals = tl.load(hidden_states_ptr + hs_offsets, mask=hs_mask, other=0.0)
                    
                    dot_prod += tl.sum(hs_vals * weight_vals)
                
                bias_val = tl.load(bias_ptr + col_idx)
                dot_prod += bias_val
                
                out_offsets = head_idx * stride_b + seq_idx * stride_s_out + dim_idx
                tl.store(output_ptr + out_offsets, dot_prod)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match the BERT pattern with 2 heads:
    linear + view(..., 2, 64) + transpose
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.linear(in_3, tmp_1, tmp_0)
    tmp_3 = tmp_2.view(in_3.size(0), -1, 2, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_2, tmp_3, tmp_4


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_3)


@torch.fx.wrap
def fused_linear_view_transpose_bert(in_0, in_1, in_3):
    """
    Fused Linear + View + Transpose for BERT (hidden_dim=128, 2 heads).
    """
    batch_size = in_3.shape[0]
    seq_len = in_3.shape[1]
    hidden_dim = in_3.shape[2]
    
    # BERT: 2 heads, 64 dim
    num_heads = 2
    head_dim = 64
    
    output = torch.empty((batch_size, num_heads, seq_len, head_dim), 
                         device=in_3.device, dtype=in_3.dtype)
    
    grid = (batch_size,)
    
    fused_linear_view_transpose_bert_kernel[grid](
        in_3, in_1, in_0,
        output,
        batch_size, seq_len, hidden_dim,
        num_heads, head_dim,
        in_3.stride(0), in_3.stride(1), in_3.stride(2),
        in_1.stride(0), in_1.stride(1),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    
    return output


def replacement_func():
    return fused_linear_view_transpose_bert