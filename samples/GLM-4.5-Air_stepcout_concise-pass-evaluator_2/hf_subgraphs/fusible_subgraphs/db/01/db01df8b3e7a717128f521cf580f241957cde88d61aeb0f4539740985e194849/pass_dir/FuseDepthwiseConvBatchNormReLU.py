import torch
import triton
import triton.language as tl

# Pattern matching function - simplest possible relu pattern
def pattern(x):
    """Simple ReLU pattern for testing"""
    return torch.nn.functional.relu(x, inplace=False)

# Argument extraction function
def replacement_args(x):
    return (x,)

@triton.jit
def triton_relu_kernel(
    input_ptr,
    output_ptr,
    num_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Simple ReLU kernel"""
    pid = tl.program_id(0)
    
    # Each program processes a block of elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Load input data
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU
    relu_vals = tl.maximum(input_vals, 0.0)
    
    # Store the result
    tl.store(output_ptr + offsets, relu_vals, mask=mask)

@triton.jit
def fused_bn_relu_kernel(
    input_ptr,
    running_mean_ptr, running_var_ptr, weight_ptr_bn, bias_ptr_bn,
    output_ptr,
    N, C, H, W,
    momentum: tl.constexpr, eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Fused batch normalization and ReLU kernel"""
    pid = tl.program_id(0)
    num_elements = N * C * H * W
    
    # Each program processes a block of elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Load input data
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Process each element in the block
    for i in range(BLOCK_SIZE):
        if offsets[i] < num_elements:
            # Calculate channel index for this position
            linear_idx = offsets[i]
            c = (linear_idx % (H * W)) // (H * W) * C + (linear_idx // (H * W)) % C
            
            # Load batch norm parameters for this channel
            mean_val = tl.load(running_mean_ptr + c).to(tl.float32)
            var_val = tl.load(running_var_ptr + c).to(tl.float32)
            weight_bn = tl.load(weight_ptr_bn + c).to(tl.float32)
            bias_bn = tl.load(bias_ptr_bn + c).to(tl.float32)
            
            # Apply batch normalization
            normalized = (input_vals[i] - mean_val) * weight_bn * tl.rsqrt(var_val + eps) + bias_bn
            
            # Apply ReLU
            relu_val = tl.maximum(normalized, 0.0)
            
            # Store the result
            tl.store(output_ptr + offsets[i], relu_val.to(output_ptr.dtype.element_ty), mask=mask)

@torch.fx.wrap
def triton_relu_optimized(x):
    """Simple optimized ReLU using Triton"""
    N, C, H, W = x.shape
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Block size for efficient processing
    BLOCK_SIZE = 1024
    
    # Number of programs needed
    total_elements = N * C * H * W
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    triton_relu_kernel[(num_programs,)](
        x, 
        output,
        total_elements,
        BLOCK_SIZE
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return triton_relu_optimized