import torch
import triton
import triton.language as tl

# Pattern matching function - matches the reshape sequence
def pattern(in_0, in_1):
    # Match the exact computation from the model
    tmp_0 = torch.nn.functional.softmax(in_0, dim=-1)
    tmp_1 = torch.nn.functional.dropout(tmp_0, p=0.0, training=False)
    tmp_2 = torch.bmm(tmp_1, in_1)
    tmp_3 = tmp_2.view(1, -1, 1, -1)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.reshape(1, 1, -1)
    return tmp_5

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel that directly computes the final output shape
@triton.jit
def optimized_bmm_reshape_kernel(
    attn_weights_ptr,
    value_states_ptr,
    output_ptr,
    batch_size,
    value_dim,
    TOTAL_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute attention using softmax + bmm directly
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < TOTAL_ELEMENTS
    
    # Reshape input for batch processing
    attn_weights_2d = tl.load(attn_weights_ptr + offset // value_dim, 
                              mask=(offset // value_dim) < batch_size, 
                              other=0.0).to(tl.float32)
    
    value_states_2d = tl.load(value_states_ptr + offset, 
                              mask=mask, 
                              other=0.0).to(tl.float32)
    
    # Softmax on attention weights
    max_attn = tl.max(attn_weights_2d, axis=0)
    exp_attn = tl.exp(attn_weights_2d - max_attn)
    sum_exp = tl.sum(exp_attn, axis=0)
    softmax_weights = exp_attn / sum_exp
    
    # Apply attention to value states and accumulate
    # Reshape for broadcasting: softmax_weights [batch_size, 1] * value_states_2d [batch_size, value_dim]
    weighted_values = softmax_weights[:, None] * value_states_2d
    
    # Sum over batch dimension to get final output
    if offset[0] < value_dim:  # Only process each column once
        output_slice = tl.sum(weighted_values[:, offset % value_dim], axis=0)
        tl.store(output_ptr + offset, output_slice, mask=offset[0] < value_dim)

@torch.fx.wrap
def optimized_bmm_reshape_forward(in_0, in_1):
    batch_size, _, _ = in_0.shape
    value_dim = in_1.shape[-1]
    total_elements = batch_size * value_dim
    
    # Direct reshape to final output shape [1, 1, batch_size * value_dim]
    output_shape = (1, 1, total_elements)
    output = torch.zeros(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Grid configuration for Triton kernel
    BLOCK_SIZE = 1024  # Process large chunks efficiently
    grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_bmm_reshape_kernel[grid](
        in_0,
        in_1,
        output,
        batch_size,
        value_dim,
        total_elements,
        BLOCK_SIZE,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_bmm_reshape_forward