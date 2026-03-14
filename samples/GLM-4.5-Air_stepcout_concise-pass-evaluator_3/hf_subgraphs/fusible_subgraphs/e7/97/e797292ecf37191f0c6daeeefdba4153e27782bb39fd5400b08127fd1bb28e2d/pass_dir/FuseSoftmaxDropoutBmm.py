import torch
import triton
import triton.language as tl

# Pattern matching function - matches the complete softmax->dropout->bmm sequence
def pattern(in_0, in_1):
    # Match the complete computation chain that produces the final output
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

# Triton kernel that fuses softmax + dropout (no-op) + bmm
@triton.jit
def fused_attention_kernel(
    attn_weights_ptr,
    value_states_ptr,
    output_ptr,
    batch_size,
    value_dim,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Process elements in blocks for better GPU utilization
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < total_elements
    
    # Load attention weights [batch_size, 1, 1] - reshape to [batch_size, 1] for easier processing
    attn_weights = tl.load(attn_weights_ptr + offset // value_dim, 
                           mask=(offset // value_dim) < batch_size, 
                           other=0.0).to(tl.float32)
    
    # Load value states [batch_size, 1, value_dim]
    value_states = tl.load(value_states_ptr + offset, 
                           mask=mask, 
                           other=0.0).to(tl.float32)
    
    # Apply softmax to attention weights (dropout with p=0.0 is no-op)
    # For attention weights [batch_size, 1], apply softmax along the last dimension
    max_attn = tl.max(attn_weights, axis=0)
    exp_attn = tl.exp(attn_weights - max_attn)
    sum_exp = tl.sum(exp_attn, axis=0)
    softmax_weights = exp_attn / sum_exp
    
    # Apply attention to value states using broadcasting
    # softmax_weights [batch_size, 1] * value_states [batch_size, value_dim] -> [batch_size, value_dim]
    weighted_values = softmax_weights[:, None] * value_states
    
    # For each output position, accumulate the weighted values from all batch members
    if offset[0] < value_dim:  # Each output column process once
        output_col = tl.sum(weighted_values[:, offset % value_dim], axis=0)
        tl.store(output_ptr + offset, output_col, mask=offset[0] < value_dim)

@torch.fx.wrap  
def fused_attention_forward(in_0, in_1):
    batch_size = in_0.shape[0]  # batch dimension from [batch_size, 1, 1]
    value_dim = in_1.shape[-1]   # value dimension from [batch_size, 1, value_dim]
    
    # Output shape: [1, 1, batch_size * value_dim]
    total_elements = batch_size * value_dim
    output = torch.zeros((1, 1, total_elements), dtype=in_0.dtype, device=in_0.device)
    
    # Triton kernel launch configuration
    BLOCK_SIZE = 512  # Process moderate chunks for good balance
    grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_attention_kernel[grid](
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
    return fused_attention_forward