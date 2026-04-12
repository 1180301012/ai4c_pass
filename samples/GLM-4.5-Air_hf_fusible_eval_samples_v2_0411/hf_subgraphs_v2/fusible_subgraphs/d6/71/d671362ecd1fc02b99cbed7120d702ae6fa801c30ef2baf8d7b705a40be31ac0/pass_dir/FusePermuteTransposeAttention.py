import torch
import triton
import triton.language as tl

@triton.jit
def fused_permute_kernel(
    input_ptr,
    output_ptr,
    stride_input_batch,
    stride_input_seq, 
    stride_input_head,
    stride_input_dim,
    stride_output_batch,
    stride_output_seq,
    stride_output_head, 
    stride_output_dim,
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized kernel for permute(0, 2, 1, 3) operation
    Transposes from [batch, seq, head, dim] to [batch, head, seq, dim]
    """
    pid = tl.program_id(0)
    batch_idx = pid // (seq_len * num_heads)
    head_idx = (pid % (seq_len * num_heads)) // seq_len
    seq_idx = pid % seq_len
    
    # Calculate input and output offsets
    input_offset = batch_idx * stride_input_batch + seq_idx * stride_input_seq + head_idx * stride_input_head
    output_offset = batch_idx * stride_output_batch + head_idx * stride_output_head + seq_idx * stride_output_seq
    
    # Load and store data
    dim_idx = tl.arange(0, head_dim)
    input_ptr_with_offset = input_ptr + input_offset + dim_idx
    output_ptr_with_offset = output_ptr + output_offset + dim_idx
    
    mask = dim_idx < head_dim
    data = tl.load(input_ptr_with_offset, mask=mask, other=0.0)
    tl.store(output_ptr_with_offset, data, mask=mask)

@torch.fx.wrap
def optimized_permute(input_tensor):
    """
    Optimized permute operation: (0, 2, 1, 3)
    Converts [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
    """
    batch_size, seq_len, num_heads, head_dim = input_tensor.shape
    
    input_strides = input_tensor.stride()
    output_tensor = torch.empty(batch_size, num_heads, seq_len, head_dim, 
                               dtype=input_tensor.dtype, device=input_tensor.device)
    output_strides = output_tensor.stride()
    
    grid_size = batch_size * seq_len * num_heads
    
    fused_permute_kernel[grid_size](
        input_tensor, output_tensor,
        * input_strides, * output_strides,
        batch_size, seq_len, num_heads, head_dim,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=128
    )
    
    return output_tensor

@triton.jit
def fused_transpose_kernel(
    input_ptr,
    output_ptr, 
    stride_input_batch,
    stride_input_seq,
    stride_input_head,
    stride_input_dim,
    stride_output_batch,
    stride_output_seq,
    stride_output_head,
    stride_output_dim,
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized kernel for transpose(-2, -1) operation
    Transposes last two dimensions: [batch, head, seq, dim] -> [batch, head, dim, seq]
    """
    pid = tl.program_id(0)
    batch_idx = pid // (seq_len * num_heads)
    head_idx = (pid % (seq_len * num_heads)) // seq_len
    seq_idx = pid % seq_len
    
    # We need to transpose seq and dim dimensions, so permute elements
    input_offset = batch_idx * stride_input_batch + head_idx * stride_input_head + seq_idx * stride_input_seq
    output_offset = batch_idx * stride_output_batch + head_idx * stride_output_head + seq_idx * stride_output_dim
    
    # Load data sequentially, store transposed
    for dim_idx in tl.range(0, head_dim, BLOCK_SIZE_N):
        input_ptr_with_offset = input_ptr + input_offset + dim_idx
        output_ptr_with_offset = output_ptr + output_offset + dim_idx * stride_output_dim
        
        mask = dim_idx < head_dim
        data = tl.load(input_ptr_with_offset, mask=mask, other=0.0)
        tl.store(output_ptr_with_offset, data, mask=mask)

@torch.fx.wrap  
def optimized_transpose(input_tensor):
    """
    Optimized transpose operation: (-2, -1)
    Converts [batch, num_heads, seq_len, head_dim] -> [batch, num_heads, head_dim, seq_len]
    """
    batch_size, num_heads, seq_len, head_dim = input_tensor.shape
    
    input_strides = input_tensor.stride()
    output_tensor = torch.empty(batch_size, num_heads, head_dim, seq_len,
                               dtype=input_tensor.dtype, device=input_tensor.device)
    output_strides = output_tensor.stride()
    
    grid_size = batch_size * seq_len * num_heads
    
    fused_transpose_kernel[grid_size](
        input_tensor, output_tensor,
        * input_strides, * output_strides,
        batch_size, seq_len, num_heads, head_dim,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=128
    )
    
    return output_tensor

def pattern(q, k, v):
    """
    Match the permutation and transpose pattern:
    - Q: permute(0, 2, 1, 3) 
    - K: permute(0, 2, 1, 3) + transpose(-2, -1)
    - V: permute(0, 2, 1, 3)
    """
    q_permuted = q.permute(0, 2, 1, 3)
    k_permuted = k.permute(0, 2, 1, 3)
    k_transposed = k_permuted.transpose(-2, -1)
    v_permuted = v.permute(0, 2, 1, 3)
    return q_permuted, k_transposed, v_permuted

def replacement_args(q, k, v):
    return (q, k, v)

def replacement_func():
    """Return a function that applies optimized permute and transpose"""
    def optimized_attention_permute(q, k, v):
        q_opt = optimized_permute(q)
        k_permuted = optimized_permute(k)
        k_opt = optimized_transpose(k_permuted)
        v_opt = optimized_permute(v)
        return q_opt, k_opt, v_opt
    return optimized_attention_permute