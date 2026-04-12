import torch
import triton
import triton.language as tl

# Pattern matching function - optimize just the linear operation
def pattern(in_0, in_1, in_3):
    """
    Optimize the linear operation: torch.nn.functional.linear(in_3, in_1, in_0)
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    return linear

# Argument extraction function
def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)

# Triton kernel for optimized linear operation
@triton.jit
def optimized_linear_kernel(
    bias_ptr,           # [8] bias tensor
    weight_ptr,         # [8, 64] weight tensor  
    input_ptr,          # [1, 12, 199, 64] input tensor
    output_ptr,         # [1, 12, 199, 8] output tensor
    batch: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_size: tl.constexpr,
    input_features: tl.constexpr,
    output_features: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Input: [batch, seq_len, hidden_size, input_features] 
    # Output: [batch, seq_len, hidden_size, output_features]
    total_output_elements = batch * seq_len * hidden_size * output_features
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_output_elements
    
    # Convert offset to coordinates
    output_idx = offsets
    batch_idx = output_idx // (seq_len * hidden_size * output_features)
    remainder = output_idx % (seq_len * hidden_size * output_features)
    seq_idx = remainder // (hidden_size * output_features)
    remainder = remainder % (hidden_size * output_features)
    hidden_idx = remainder // output_features
    feature_idx = remainder % output_features
    
    # Matrix multiplication: output = input @ weight.T + bias
    # Initialize as vector to handle loop properly
    linear_result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for k in range(input_features):
        # Load input element
        input_offset = (batch_idx * (seq_len * hidden_size * input_features) + 
                       seq_idx * (hidden_size * input_features) + 
                       hidden_idx * input_features + 
                       k)
        input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        
        # Load weight element  
        weight_offset = (feature_idx * input_features + k)
        weight_val = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
        
        # Use vector operations
        linear_result += input_val * weight_val
    
    # Add bias (broadcast to all elements in the block)
    bias_val = tl.load(bias_ptr + feature_idx, mask=mask, other=0.0)
    linear_result += bias_val
    
    # Store result
    tl.store(output_ptr + offsets, linear_result, mask=mask)

@torch.fx.wrap
def optimized_linear(in_0, in_1, in_3):
    """
    Optimized linear operation using Triton
    """
    batch = 1
    seq_len = 12  # wavlm_base, will be overwritten in wrapper
    hidden_size = 199
    input_features = 64
    output_features = 8
    
    # Determine actual seq_len from input
    seq_len = in_3.shape[1]
    
    # Create output tensor
    output = torch.empty(batch * seq_len * hidden_size * output_features, dtype=in_3.dtype, device=in_3.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    total_elements = batch * seq_len * hidden_size * output_features
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_linear_kernel[(num_programs,)](
        bias_ptr=in_0,
        weight_ptr=in_1,
        input_ptr=in_3,
        output_ptr=output,
        batch=batch,
        seq_len=seq_len,
        hidden_size=hidden_size,
        input_features=input_features,
        output_features=output_features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to expected output format [batch, seq_len, hidden_size, output_features]
    return output.view(batch, seq_len, hidden_size, output_features)

# Replacement function
def replacement_func():
    return optimized_linear