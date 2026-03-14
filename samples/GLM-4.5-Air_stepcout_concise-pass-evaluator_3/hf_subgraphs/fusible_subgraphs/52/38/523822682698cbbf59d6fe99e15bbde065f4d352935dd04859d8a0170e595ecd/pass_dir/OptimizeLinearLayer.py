import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_6, tmp_5, tmp_4):
    """ Match the linear operation: in_6 @ tmp_5.T + tmp_4 """
    tmp_6 = torch.nn.functional.linear(in_6, tmp_5, tmp_4)
    return tmp_6

# Argument extraction function
def replacement_args(in_6, tmp_5, tmp_4):
    return (in_6, tmp_5, tmp_4)

# Optimized kernel for linear layer
@triton.jit
def linear_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    input_features: tl.constexpr,
    output_features: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """ Simple Triton kernel for linear layer with constexpr block size """
    pid_batch = tl.program_id(0)
    pid_output = tl.program_id(1)
    
    # Check bounds
    if pid_batch >= batch_size or pid_output >= output_features:
        return
    
    # Create ranges using power-of-2 block sizes (BLOCK_SIZE_K is now constexpr)
    in_offsets = tl.arange(0, BLOCK_SIZE_K)
    out_idx = pid_output  # Single output feature per program
    
    # Mask for input bounds
    in_mask = in_offsets < input_features
    
    # Load input vector for this batch element
    x_ptr_batch = x_ptr + pid_batch * input_features
    x = tl.load(x_ptr_batch + in_offsets, mask=in_mask, other=0.0)
    
    # Load weights for this single output feature
    w_ptr_row = w_ptr + out_idx * input_features
    w = tl.load(w_ptr_row + in_offsets, mask=in_mask, other=0.0)
    
    # Load bias for this output feature
    b = tl.load(b_ptr + out_idx)  # No mask needed for single element load
    
    # Matrix multiplication: dot product of x and w
    acc = tl.sum(x * w)
    
    # Add bias
    out = acc + b
    
    # Store result
    out_ptr_base = out_ptr + pid_batch * output_features + out_idx
    tl.store(out_ptr_base, out)

# Kernel wrapper
@torch.fx.wrap
def optimized_linear(in_6, tmp_5, tmp_4):
    """ Optimized linear layer using Triton """
    batch_size, input_features = in_6.shape
    output_features = tmp_4.shape[0]
    
    # Use 2D grid: one program per (batch, output_feature) pair
    grid_m = batch_size
    grid_n = output_features
    
    # Power-of-2 block size for input features
    BLOCK_SIZE_K = 256
    
    # Allocate output tensor
    out = torch.empty((batch_size, output_features), dtype=in_6.dtype, device=in_6.device)
    
    # Launch kernel with 2D grid and constexpr parameter
    linear_kernel[(grid_m, grid_n)](
        x_ptr=in_6,
        w_ptr=tmp_5,
        b_ptr=tmp_4,
        out_ptr=out,
        batch_size=batch_size,
        input_features=input_features,
        output_features=output_features,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

# Replacement function
def replacement_func():
    return optimized_linear