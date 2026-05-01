import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches the exact sequence of operations in the computation graph
# with the specific tensor shapes from the target graph

@torch.fx.wrap
def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_4 = linear.view(1, 12, 199, 2, 4)
    tmp_5 = tmp_4.sum(-1, keepdim=False)
    tmp_6 = torch.sigmoid(tmp_5)
    chunk = tmp_6.chunk(2, dim=-1)
    tmp_8 = chunk[0]
    tmp_9 = chunk[1]
    tmp_10 = tmp_9 * in_2
    tmp_11 = tmp_10 - 1.0
    tmp_12 = tmp_8 * tmp_11
    tmp_13 = tmp_12 + 2.0
    tmp_14 = tmp_13.view(1, 12, -1, 1)
    return tmp_14

# Argument extraction function
# Extracts the necessary input tensors

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Optimized kernel implementing the entire sequence in one Triton kernel
@triton.jit
def fused_kernel(
    in_3_ptr, 
    in_1_ptr, 
    in_0_ptr, 
    in_2_ptr, 
    out_ptr, 
    num_heads, 
    num_time_steps, 
    n_features, 
    n_output_features, 
    BLOCK_SIZE: tl.constexpr):
    
    # Each thread processes one head and time step
    head_idx = tl.program_id(0)
    time_idx = tl.program_id(1)
    
    # Load bias values for all output features
    bias = tl.load(in_0_ptr + tl.arange(0, n_output_features), dtype=tl.float32)
    
    # Initialize sums for the two groups (first 4 and last 4 output features)
    sum0 = tl.zeros((1,), dtype=tl.float32)
    sum1 = tl.zeros((1,), dtype=tl.float32)
    
    # Process all input features (64)
    for m in range(n_features):
        # Load input value for current position
        input_val = tl.load(
            in_3_ptr + (head_idx * num_time_steps * n_features + time_idx * n_features + m),
            dtype=tl.float32
        )
        
        # Load weights for the current input feature column
        weights_m = tl.load(
            in_1_ptr + (tl.arange(0, n_output_features) * n_features + m),
            dtype=tl.float32
        )
        
        # Compute linear output for all 8 features
        linear_out = input_val * weights_m + bias
        
        # Accumulate into sum0 (first 4 features) and sum1 (last 4 features)
        sum0 += linear_out[0] + linear_out[1] + linear_out[2] + linear_out[3]
        sum1 += linear_out[4] + linear_out[5] + linear_out[6] + linear_out[7]

    # Apply sigmoid to the accumulated sums
    sig0 = 1.0 / (1.0 + tl.exp(-sum0))
    sig1 = 1.0 / (1.0 + tl.exp(-sum1))
    
    # Load constant value for this head
    const_val = tl.load(in_2_ptr + (head_idx * 1), dtype=tl.float32)
    
    # Compute final result: (sig1 * const - 1) * sig0 + 2
    result = sig0 * (sig1 * const_val - 1.0) + 2.0
    
    # Store result in output tensor
    tl.store(out_ptr + (head_idx * num_time_steps + time_idx), result, dtype=tl.float32)

# Kernel wrapper with proper grid setup
@torch.fx.wrap
def fused_linear_processing_wrapper(in_0, in_1, in_2, in_3):
    # Get dimensions from input tensor
    _, num_heads, num_time_steps, _ = in_3.shape
    
    # Create output tensor with correct shape
    out = torch.empty(1, num_heads, num_time_steps, 1, dtype=in_3.dtype, device=in_3.device)
    
    # Launch kernel with grid matching heads and time steps
    grid = (num_heads, num_time_steps)
    
    # Call the Triton kernel
    fused_kernel[grid](
        in_3_ptr=in_3,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        in_2_ptr=in_2,
        out_ptr=out,
        num_heads=num_heads,
        num_time_steps=num_time_steps,
        n_features=64,  # Input feature dimension
        n_output_features=8,  # Output feature dimension
        BLOCK_SIZE=1
    )
    
    return out

# Replacement function (returns the optimized implementation)
def replacement_func():
    return fused_linear_processing_wrapper