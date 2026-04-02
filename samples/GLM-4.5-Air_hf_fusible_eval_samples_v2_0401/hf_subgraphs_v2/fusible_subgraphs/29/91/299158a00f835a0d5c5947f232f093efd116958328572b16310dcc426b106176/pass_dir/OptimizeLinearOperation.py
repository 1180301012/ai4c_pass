import torch
import triton
import triton.language as tl

def pattern(input, weight, bias):
    """
    Pattern matching for torch.nn.functional.linear operation
    """
    result = torch.nn.functional.linear(input, weight, bias)
    return result

def replacement_args(input, weight, bias):
    """
    Extract arguments for the optimized linear operation
    """
    return (input, weight, bias)

@triton.jit
def linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_batch, input_features, output_features, input_dtype: tl.constexpr
):
    """
    Optimized Triton kernel for linear operation: output = input @ weight.t() + bias
    Simple and efficient single-element computation
    """
    # Program id for each output element
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Check bounds
    mask_m = m < input_batch
    mask_n = n < output_features
    
    # Early return if out of bounds
    if not (mask_m and mask_n):
        return
    
    # Load bias for this output element
    bias_val = tl.load(bias_ptr + n, mask=mask_n, other=0.0).to(tl.float32)
    
    # Initialize accumulator with bias
    accumulator = bias_val
    
    # Compute dot product: input_row @ weight_col
    for k in range(input_features):
        # Load input element (m, k)
        input_val = tl.load(
            input_ptr + m * input_features + k,
            mask=mask_m,
            other=0.0
        ).to(tl.float32)
        
        # Load weight element (n, k)
        weight_val = tl.load(
            weight_ptr + n * input_features + k
        ).to(tl.float32)
        
        # Accumulate dot product
        accumulator += input_val * weight_val
    
    # Determine output data type and convert
    if input_dtype == 0:  # float32
        output = accumulator.to(tl.float32)
    elif input_dtype == 1:  # float16
        output = accumulator.to(tl.float16)
    else:  # bfloat16
        output = accumulator.to(tl.bfloat16)
    
    # Store result
    tl.store(output_ptr + m * output_features + n, output)

@torch.fx.wrap
def optimized_linear(input, weight, bias):
    """
    Optimized linear operation using Triton kernel
    """
    # Get input shapes
    input_batch, input_features = input.shape
    output_features = weight.shape[0]
    
    # Calculate output shape
    output_shape = (input_batch, output_features)
    
    # Create output tensor
    if input.dtype == torch.float32:
        output = torch.empty(output_shape, dtype=torch.float32, device=input.device)
    elif input.dtype == torch.float16:
        output = torch.empty(output_shape, dtype=torch.float16, device=input.device)
    else:  # bfloat16
        output = torch.empty(output_shape, dtype=torch.bfloat16, device=input.device)
    
    # Map data types to integers
    if input.dtype == torch.float32:
        dtype_int = 0
    elif input.dtype == torch.float16:
        dtype_int = 1
    else:  # bfloat16
        dtype_int = 2
    
    # Grid dimensions: one launch per output element
    grid_m = input_batch
    grid_n = output_features
    
    # Launch kernel
    linear_kernel[(grid_m, grid_n)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        input_batch=input_batch,
        input_features=input_features,
        output_features=output_features,
        input_dtype=dtype_int
    )
    
    return output

def replacement_func():
    """
    Returns the optimized linear function
    """
    return optimized_linear