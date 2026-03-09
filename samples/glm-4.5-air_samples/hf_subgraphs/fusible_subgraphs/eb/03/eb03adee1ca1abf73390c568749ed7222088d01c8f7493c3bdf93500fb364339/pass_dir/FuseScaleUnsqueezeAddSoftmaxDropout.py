import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """Match the computation pattern: scalar_mul + unsqueeze + add + softmax + dropout(p=0)"""
    # Step 1: scalar multiplication
    tmp_0 = in_1 * 0.1767766952966369
    # Step 2: unsqueeze
    tmp_1 = in_0.unsqueeze(2)
    # Step 3: broadcasting add
    tmp_2 = tmp_0 + tmp_1
    # Step 4: softmax along last dimension
    tmp_3 = tmp_2.softmax(dim=-1)
    # Step 5: dropout with p=0.0 (no-op, but still in the graph)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4


def replacement_args(in_0, in_1):
    """Extract arguments needed for the replacement"""
    return (in_0, in_1)


# Using grid = batch * num_seqs * num_heads (1083 programs)
# Each program handles one (b,s,h) and computes 49 softmaxes (one for each x)
@triton.jit
def fused_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    N: tl.constexpr,
    num_seqs: tl.constexpr, 
    num_heads: tl.constexpr,
    stride_in_0_b, stride_in_0_s,
    stride_in_1_b, stride_in_1_s, stride_in_1_h,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Decode to batch, seq, head
    batch_idx = pid // (num_seqs * num_heads)
    remainder = pid % (num_seqs * num_heads)
    seq_idx = remainder // num_heads
    head_idx = remainder % num_heads
    
    # For in_0[b, s, x, y], we load all x rows (49 of them)
    # For in_1[b, s, h, x, y], we load for this head (h)
    # Each program computes 49 softmaxes (for x = 0..48)
    
    # Offsets for in_0 base (batch, seq)
    in_0_base = batch_idx * stride_in_0_b + seq_idx * stride_in_0_s
    
    # Offset for in_1 (batch, seq, head)
    in_1_base = batch_idx * stride_in_1_b + seq_idx * stride_in_1_s + head_idx * stride_in_1_h
    
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Process all x values (0 to 48) - 49 iterations
    # For each x, load in_0[b,s,x,:] and in_1[b,s,h,x,:], add, softmax, store
    for x_idx in range(N):
        # in_0[b,s,x,:] - load 49 values
        in_0_offset = in_0_base + x_idx * N
        in_0_ptrs = in_0_ptr + in_0_offset + offs
        in_0_vals = tl.load(in_0_ptrs, mask=mask, other=0.0)
        
        # in_1[b,s,h,x,:] - load 49 values  
        in_1_offset = in_1_base + x_idx * N
        in_1_ptrs = in_1_ptr + in_1_offset + offs
        in_1_vals = tl.load(in_1_ptrs, mask=mask, other=0.0)
        
        # Scale, add, softmax
        combined = in_1_vals * scale + in_0_vals
        
        max_val = tl.max(combined, axis=0)
        max_val = tl.broadcast_to(max_val, (BLOCK_SIZE,))
        shifted = combined - max_val
        exp_shifted = tl.exp(shifted)
        sum_exp = tl.sum(exp_shifted, axis=0)
        softmax_out = exp_shifted / sum_exp
        
        # Store result at in_1 position
        out_ptrs = out_ptr + in_1_offset + offs
        tl.store(out_ptrs, softmax_out, mask=mask)


@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    scale = 0.1767766952966369
    batch_size = in_0.shape[0]
    num_seqs = in_0.shape[1]
    x_size = in_0.shape[2]  # 49
    y_size = in_0.shape[3]  # 49 (softmax dim)
    num_heads = in_1.shape[2]
    
    total_rows = batch_size * num_seqs * num_heads
    output = torch.empty_like(in_1)
    
    stride_in_0_b = in_0.stride(0)
    stride_in_0_s = in_0.stride(1)
    stride_in_1_b = in_1.stride(0)
    stride_in_1_s = in_1.stride(1)
    stride_in_1_h = in_1.stride(2)
    
    grid = (total_rows,)
    BLOCK_SIZE = 64
    
    fused_kernel[grid](
        in_0_ptr=in_0, in_1_ptr=in_1, out_ptr=output,
        N=x_size, num_seqs=num_seqs, num_heads=num_heads,
        stride_in_0_b=stride_in_0_b, stride_in_0_s=stride_in_0_s,
        stride_in_1_b=stride_in_1_b, stride_in_1_s=stride_in_1_s, stride_in_1_h=stride_in_1_h,
        scale=scale, BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def replacement_func():
    return kernel_wrapper