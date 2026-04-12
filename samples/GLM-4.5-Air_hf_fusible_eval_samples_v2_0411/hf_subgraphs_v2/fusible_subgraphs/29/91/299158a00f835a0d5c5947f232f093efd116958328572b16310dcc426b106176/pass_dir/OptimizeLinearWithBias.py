import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_2, in_1, in_0):
    """Pattern for torch.nn.functional.linear(in_2, in_1, in_0)"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
    return tmp_2

# Argument extraction function
def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

# Triton kernel for optimized linear operation
@triton.jit  
def linear_kernel(
    x_ptr, x_stride,
    w_ptr, w_stride, 
    b_ptr,
    y_ptr, y_stride,
    batch_size, out_features, in_features,
    BLOCK_SIZE: tl.constexpr
):
    # Each program computes one output element
    pid = tl.program_id(0)
    
    # Determine which output element this program computes
    row = pid // out_features
    col = pid % out_features
    
    # Check if this position is valid
    if row >= batch_size or col >= out_features:
        return
    
    # Compute dot product: x[row, :] @ w[col, :] + b[col]
    acc = 0.0
    for k in range(0, in_features, BLOCK_SIZE):
        # Create range for current block
        offsets = k + tl.arange(0, BLOCK_SIZE)
        mask = offsets < in_features
        
        # Load elements from x and w
        x_block = tl.load(
            x_ptr + row * x_stride + offsets,
            mask=mask,
            other=0.0
        )
        w_block = tl.load(
            w_ptr + col * w_stride + offsets,
            mask=mask,
            other=0.0
        )
        
        # Compute partial dot product
        acc += tl.sum(x_block * w_block)
    
    # Load bias and add (bias only has one dimension)
    col_mask = col < out_features
    b_val = tl.load(b_ptr + col, mask=col_mask, other=0.0)
    acc += b_val
    
    # Store result
    tl.store(y_ptr + row * y_stride + col, acc)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_linear(input_tensor, weight_tensor, bias_tensor):
    # Get tensor shapes
    batch_size = input_tensor.shape[0]
    out_features = weight_tensor.shape[0]  # Output features (dim 0 of weight)
    in_features = input_tensor.shape[1]    # Input features (dim 1 of input)
    
    # Determine output shape
    output_shape = (batch_size, out_features)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Use a reasonable block size for vectorized memory access
    BLOCK_SIZE = 128
    
    # Compute grid size - one program per output element
    total_elements = batch_size * out_features
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    linear_kernel[(
        grid_size,
    )](
        x_ptr=input_tensor,
        x_stride=input_tensor.stride(1),
        w_ptr=weight_tensor,
        w_stride=weight_tensor.stride(1),
        b_ptr=bias_tensor,
        y_ptr=output,
        y_stride=output.stride(1),
        batch_size=batch_size,
        out_features=out_features,
        in_features=in_features,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_linear