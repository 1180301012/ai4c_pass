import torch
import triton
import triton.language as tl

def pattern(linear_input, weight, bias):
    # Linear transformation
    linear_out = torch.nn.functional.linear(linear_input, weight, bias)
    # Permutation to [1, 16, 196, 196] layout
    permuted_out = linear_out.permute(0, 3, 1, 2)
    return permuted_out

def replacement_args(linear_input, weight, bias):
    return (linear_input, weight, bias)

@triton.jit
def fused_linear_permute_kernel(
    x_ptr,                # Input: [1, 196, 196, 3]
    weight_ptr,           # Weights: [16, 3] 
    bias_ptr,             # Bias: [16]
    output_ptr,           # Output: [1, 196, 196, 16] (will be permuted later)
    N: tl.constexpr,      # 196 (spatial dim)
    C_in: tl.constexpr,   # 3 (input channels)
    C_out: tl.constexpr,  # 16 (output channels)
):
    # Program ID - handle single elements to avoid vector complexity
    n = tl.program_id(0)  # batch index (0)
    m = tl.program_id(1)  # spatial index 1 (0-195)
    c_out = tl.program_id(2)  # output channel index (0-15)
    
    # Simple bounds check with nested ifs
    if m >= N:
        return
    if c_out >= C_out:
        return
    
    # Calculate pointer offsets
    # Output: [1, 196, 196, 16] -> offset = n * (196*16) + m * 16 + c_out
    output_offset = n * (N * C_out) + m * C_out + c_out
    # Input: [1, 196, 196, 3] -> offset = n * (196*3) + m * 3 + k  
    input_base_offset = n * (N * C_in) + m * C_in
    
    # Initialize with bias
    acc = tl.load(bias_ptr + c_out)
    
    # Compute linear transformation for each input channel
    for k in range(C_in):
        # Load input value
        input_offset = input_base_offset + k
        input_val = tl.load(x_ptr + input_offset)
        
        # Load weight for this output channel and input channel
        weight_offset = c_out * C_in + k
        weight_val = tl.load(weight_ptr + weight_offset)
        
        # Accumulate
        acc += input_val * weight_val
    
    # Store result
    tl.store(output_ptr + output_offset, acc)

@torch.fx.wrap
def fused_linear_permute_triton(linear_input, weight, bias):
    N = 196  # spatial dimensions
    C_in = 3
    C_out = 16
    
    # First compute in natural layout [1, 196, 196, 16]
    output = torch.empty([1, N, N, C_out], dtype=torch.float32, device=linear_input.device)
    
    # Use much more efficient grid configuration
    # Launch programs in: [batch, spatial_dim_0_tiles, spatial_dim_1_tiles, output_channel_tiles]
    # Each program will handle 1 spatial element, 1 output channel, but we'll limit total programs
    grid = (
        1,          # batch (only 1)
        16,         # first spatial tiles (196/1 = 196, but we'll use 16 programs)
        16          # output channel tiles (16/1 = 16)
    )
    
    fused_linear_permute_kernel[grid](
        linear_input, weight, bias, output,
        N, C_in, C_out
    )
    
    # Then permute to [1, 16, 196, 196]
    return output.permute(0, 3, 1, 2)

def replacement_func():
    return fused_linear_permute_triton