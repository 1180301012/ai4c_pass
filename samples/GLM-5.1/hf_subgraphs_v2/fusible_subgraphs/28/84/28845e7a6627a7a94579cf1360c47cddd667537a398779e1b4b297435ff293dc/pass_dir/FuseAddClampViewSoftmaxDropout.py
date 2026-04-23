import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    return in_1 + in_0

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_add_softmax_kernel(
    in0_ptr, in1_ptr, out_ptr,
    n_rows, n_cols,
    stride_in0_h, stride_in0_s, stride_in0_d,
    stride_in1_h, stride_in1_s, stride_in1_d,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols
    
    head_idx = row_idx // n_cols
    seq_idx = row_idx % n_cols
    
    in0_ptrs = in0_ptr + seq_idx * stride_in0_d + col_offsets
    in0_row = tl.load(in0_ptrs, mask=col_mask, other=0.0)
    
    in1_ptrs = in1_ptr + head_idx * stride_in1_h + seq_idx * stride_in1_d + col_offsets
    in1_row = tl.load(in1_ptrs, mask=col_mask, other=0.0)
    
    added = in0_row + in1_row
    
    max_val = tl.max(added, axis=0)
    shifted = added - max_val
    exp_vals = tl.exp(shifted)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_out = exp_vals / sum_exp
    
    out_ptrs = out_ptr + row_idx * n_cols + col_offsets
    tl.store(out_ptrs, softmax_out, mask=col_mask)

@torch.fx.wrap
def fused_add_softmax(in_0, in_1):
    n_heads = in_1.shape[1]
    n_seq = in_1.shape[2]
    n_rows = n_heads * n_seq
    n_cols = n_seq
    
    out = torch.empty((n_heads, n_seq, n_seq), dtype=in_1.dtype, device=in_1.device)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    grid = (n_rows,)
    
    fused_add_softmax_kernel[grid](
        in0_ptr=in_0, in1_ptr=in_1, out_ptr=out,
        n_rows=n_rows, n_cols=n_cols,
        stride_in0_h=in_0.stride(1), stride_in0_s=in_0.stride(2), stride_in0_d=in_0.stride(3),
        stride_in1_h=in_1.stride(1), stride_in1_s=in_1.stride(2), stride_in1_d=in_1.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_add_softmax