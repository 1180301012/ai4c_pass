import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(input_tensor, weight, bias, eps):
    """
    Match layer normalization operation
    """
    tmp_11 = torch.nn.functional.layer_norm(input_tensor, (256,), weight, bias, eps)
    return tmp_11

# Argument extraction function
def replacement_args(input_tensor, weight, bias, eps):
    return (input_tensor, weight, bias, eps)

# Triton kernel for layer normalization
@triton.jit
def layer_norm_kernel(
    X_ptr,  # input tensor
    W_ptr,  # weight
    B_ptr,  # bias
    Y_ptr,  # output
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data block
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean
    block_sum = tl.sum(x)
    block_size = tl.sum(mask)
    mean = block_sum / block_size
    
    # Compute variance
    x_centered = x - mean
    x_centered_sq = x_centered * x_centered
    block_var_sum = tl.sum(x_centered_sq)
    var = block_var_sum / block_size + eps
    
    # Normalize
    x_norm = x_centered / tl.sqrt(var)
    
    # Load weight and bias (broadcast to feature dimension)
    weight = tl.load(W_ptr + (offsets % 256), mask=(offsets % 256) < 256, other=1.0)
    bias = tl.load(B_ptr + (offsets % 256), mask=(offsets % 256) < 256, other=0.0)
    
    # Apply affine transformation
    y = x_norm * weight + bias
    
    # Store result
    tl.store(Y_ptr + offsets, y, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_layer_norm(input_tensor, weight, bias, eps):
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Launch Triton kernel
    layer_norm_kernel[(num_programs,)](
        X_ptr=input_tensor,
        W_ptr=weight,
        B_ptr=bias,
        Y_ptr=output,
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_layer_norm