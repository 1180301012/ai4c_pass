import torch
import triton
import triton.language as tl

def pattern(input):
    """
    Pattern matching for mean operation along dimension -2
    """
    result = input.mean(-2)
    return result

def replacement_args(input):
    """
    Extract arguments for the optimized mean operation
    """
    return (input,)

@triton.jit
def mean_kernel(
    input_ptr, output_ptr,
    batch_size, seq_len, features, input_dtype: tl.constexpr
):
    """
    Optimized Triton kernel for mean operation along sequence dimension (simplified)
    """
    # Program id for each output element
    b = tl.program_id(0)
    f = tl.program_id(1)
    
    # Check bounds
    mask_b = b < batch_size
    mask_f = f < features
    
    # Early return if out of bounds
    if not (mask_b and mask_f):
        return
    
    # Initialize accumulator
    accumulator = 0.0
    
    # Sum over sequence dimension
    for s in range(seq_len):
        # Load input element (b, s, f)
        input_val = tl.load(
            input_ptr + b * seq_len * features + s * features + f,
            other=0.0
        ).to(tl.float32)
        accumulator += input_val
    
    # Compute mean
    output = accumulator / seq_len
    
    # Determine output data type
    if input_dtype == 0:  # float32
        output = output.to(tl.float32)
    elif input_dtype == 1:  # float16
        output = output.to(tl.float16)
    else:  # bfloat16
        output = output.to(tl.bfloat16)
    
    # Store result
    tl.store(output_ptr + b * features + f, output)

@torch.fx.wrap
def optimized_mean(input):
    """
    Optimized mean operation using Triton kernel
    """
    # Get input shape
    batch_size, seq_len, features = input.shape
    
    # Create output tensor with shape [batch_size, features]
    if input.dtype == torch.float32:
        output = torch.empty((batch_size, features), dtype=torch.float32, device=input.device)
    elif input.dtype == torch.float16:
        output = torch.empty((batch_size, features), dtype=torch.float16, device=input.device)
    else:  # bfloat16
        output = torch.empty((batch_size, features), dtype=torch.bfloat16, device=input.device)
    
    # Map data types to integers
    if input.dtype == torch.float32:
        dtype_int = 0
    elif input.dtype == torch.float16:
        dtype_int = 1
    else:  # bfloat16
        dtype_int = 2
    
    # Grid dimensions: one launch per output element
    grid_batch = batch_size
    grid_features = features
    
    # Launch kernel
    mean_kernel[(grid_batch, grid_features)](
        input_ptr=input,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        features=features,
        input_dtype=dtype_int
    )
    
    return output

def replacement_func():
    """
    Returns the optimized mean function
    """
    return optimized_mean