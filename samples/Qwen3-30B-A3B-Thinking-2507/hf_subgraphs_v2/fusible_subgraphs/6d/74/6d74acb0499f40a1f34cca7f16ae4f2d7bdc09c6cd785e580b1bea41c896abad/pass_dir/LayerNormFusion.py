import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2, in_3):
    x = in_3 + in_2
    x = x.float()
    mean = x.mean(-1, keepdim = True)
    x_centered = x - mean
    x_sq = x_centered.pow(2)
    var = x_sq.mean(-1, keepdim=True)
    std = torch.sqrt(var + 1e-07)
    x_norm = x_centered / std
    x_norm = x_norm.to(torch.float32)
    result = in_1 * x_norm + in_0
    return result

# Argument extraction function
# We only need the inputs, but we need to keep the same number as pattern
# because replacement_args must have the same number of parameters as pattern
# but we're not using the intermediate variables
# This is required by the framework, though we don't use them
# We use the input tensors to determine the output shape
# The framework will pass in the original inputs
# Note: We're not using the intermediate variables because they're not returned in the pattern
# But we must have the same number of parameters as pattern
# So we return (in_0, in_1, in_2, in_3)

def replacement_args(in_0, in_1, in_2, in_3):
    # Return the input tensors (we'll use these in the kernel)
    return (in_0, in_1, in_2, in_3)

# Triton kernel for fused LayerNorm
@triton.jit
def fused_layernorm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the block index
    block_idx = tl.program_id(0)
    
    # Calculate the offset within the batch and sequence dimensions
    batch_start = block_idx // seq_len
    seq_start = block_idx % seq_len
    
    # Calculate the starting index for the current block
    input_offset = (batch_start * seq_len * hidden_size) + (seq_start * hidden_size)
    output_offset = (batch_start * seq_len * hidden_size) + (seq_start * hidden_size)
    
    # Load the input data for the current position
    input = tl.load(input_ptr + input_offset, 
                   mask=tl.arange(0, hidden_size) < hidden_size,
                   other=0.0)
    weight = tl.load(weight_ptr, 
                    mask=tl.arange(0, hidden_size) < hidden_size,
                    other=0.0)
    bias = tl.load(bias_ptr, 
                  mask=tl.arange(0, hidden_size) < hidden_size,
                  other=0.0)
    
    # Calculate mean
    sum_val = tl.zeros((hidden_size,), dtype=tl.float32)
    for i in range(seq_len):
        # Calculate offset for each sequence position
        seq_offset = (batch_start * seq_len * hidden_size) + (i * hidden_size)
        seq_input = tl.load(input_ptr + seq_offset, 
                          mask=tl.arange(0, hidden_size) < hidden_size,
                          other=0.0)
        sum_val += seq_input
    mean = sum_val / seq_len
    
    # Calculate variance
    sum_sq_val = tl.zeros((hidden_size,), dtype=tl.float32)
    for i in range(seq_len):
        seq_offset = (batch_start * seq_len * hidden_size) + (i * hidden_size)
        seq_input = tl.load(input_ptr + seq_offset, 
                          mask=tl.arange(0, hidden_size) < hidden_size,
                          other=0.0)
        centered = seq_input - mean
        sum_sq_val += centered * centered
    var = sum_sq_val / seq_len
    
    # Calculate standard deviation (with epsilon)
    std = tl.sqrt(var + eps)
    
    # Normalize and scale
    normalized = (input - mean) / std
    output = normalized * weight + bias
    
    # Store the output
    tl.store(output_ptr + output_offset, output, 
            mask=tl.arange(0, hidden_size) < hidden_size)

@torch.fx.wrap
def fused_layernorm(in_0, in_1, in_2, in_3):
    # Determine the shape of the inputs
    batch_size, seq_len, hidden_size = in_3.shape
    
    # Create output tensor
    out = torch.empty_like(in_3)
    
    # Set the kernel parameters
    BLOCK_SIZE = 256
    num_blocks = (batch_size * seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    fused_layernorm_kernel[(num_blocks,)](
        in_3,  # input tensor
        in_1,  # weight tensor
        in_0,  # bias tensor
        out,   # output tensor
        batch_size,
        seq_len,
        hidden_size,
        1e-07,  # epsilon
        BLOCK_SIZE
    )
    
    return out

# Replacement function
# Return the kernel wrapper function (not called)
def replacement_func():
    return fused_layernorm