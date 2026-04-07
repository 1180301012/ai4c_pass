import torch
import triton
import triton.language as tl

def pattern(in_3, in_1, in_0):
    """Fuse linear transformation + view + sum + sigmoid + chunk operations"""
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_4 = linear.view(1, linear.shape[1], 199, 2, 4)
    tmp_5 = tmp_4.sum(-1, keepdim=False)
    tmp_6 = torch.sigmoid(tmp_5)
    chunk = tmp_6.chunk(2, dim=-1)
    tmp_8 = chunk[0]
    tmp_9 = chunk[1]
    return tmp_8, tmp_9

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@triton.jit
def fused_kernel_linear_view_sum_sigmoid_chunk(
    in_3_ptr, in_1_ptr, in_0_ptr,
    out_ptr1, out_ptr2,
    batch_size, seq_len, hidden_dim, output_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for linear + view + sum + sigmoid + chunk operations"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * seq_len)
    
    # Load input features (in_3)
    batch_seq_idx = offsets // seq_len
    within_batch_idx = offsets % seq_len
    feature_offset = tl.arange(0, hidden_dim)
    
    # Load batch * seq_len * hidden_dim values from in_3
    in_3_base_idx = batch_seq_idx * (seq_len * hidden_dim) + within_batch_idx * hidden_dim + feature_offset
    in_3_values = tl.load(in_3_ptr + in_3_base_idx, mask=mask.reshape(-1, 1), num_stages=2)
    
    # Linear transformation in chunks
    linear_results = tl.zeros((len(offsets), output_dim), dtype=tl.float32)
    for k in range(0, hidden_dim, 32):
        feature_chunk = feature_offset[k:k+32]
        in_3_chunk = tl.load(in_3_ptr + in_3_base_idx[:, None] + feature_chunk[None, :], 
                           mask=mask.reshape(-1, 1), other=0.0)
        
        weight_offset = tl.arange(0, output_dim)[:, None] * hidden_dim + feature_chunk[None, :]
        weight_chunk = tl.load(in_1_ptr + weight_offset, mask=weight_offset < (output_dim * hidden_dim))
        bias_chunk = tl.load(in_0_ptr + tl.arange(0, output_dim), mask=tl.arange(0, output_dim) < output_dim)
        
        linear_results += tl.dot(in_3_chunk.to(tl.float32), weight_chunk.to(tl.float32).T)
    linear_results += bias_chunk.to(tl.float32)
    
    # Reshape and sum operations (2*4=8 output features, summing the 4 dimension)
    linear_results = linear_results.reshape(len(offsets), output_dim // 8, 8)
    summed = tl.sum(linear_results, axis=2)  # Sum over the last dimension (4 elements)
    
    # Sigmoid activation
    activated = tl.sigmoid(summed.to(tl.float32))
    
    # Split into two chunks
    out_chunk1 = activated[:, :, 0]  # First half
    out_chunk2 = activated[:, :, 1]  # Second half
    
    # Store results
    out_base_idx = offsets.reshape(-1, 1)
    tl.store(out_ptr1 + out_base_idx, out_chunk1.reshape(-1), mask=mask)
    tl.store(out_ptr2 + out_base_idx, out_chunk2.reshape(-1), mask=mask)

@torch.fx.wrap
def fused_linear_view_sum_sigmoid_chunk(in_3, in_1, in_0):
    """Wrapper for the fused kernel"""
    batch_size, seq_len, hidden_dim = in_3.shape[0], in_3.shape[1], in_3.shape[2]
    output_dim = in_1.shape[0]
    
    # Allocate output tensors
    out_shape1 = (batch_size * seq_len, output_dim // 8)
    out1 = torch.zeros(out_shape1, dtype=in_3.dtype, device=in_3.device)
    out2 = torch.zeros(out_shape1, dtype=in_3.dtype, device=in_3.device)
    
    # Set kernel configuration
    BLOCK_SIZE = 1024
    num_programs = (batch_size * seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_kernel_linear_view_sum_sigmoid_chunk[num_programs](
        in_3_ptr=in_3,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr1=out1,
        out_ptr2=out2,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out1, out2

def replacement_func():
    return fused_linear_view_sum_sigmoid_chunk