import torch
import triton
import triton.language as tl

def pattern(tmp_3, normalized_shape, in_1, in_0, eps):
    """Pattern matches layer_norm operation"""
    result = torch.nn.functional.layer_norm(tmp_3, normalized_shape, in_1, in_0, eps)
    return result

def replacement_args(tmp_3, normalized_shape, in_1, in_0, eps):
    """Extract arguments for the optimized kernel"""
    return (tmp_3, normalized_shape, in_1, in_0, eps)

@triton.jit
def optimized_layernorm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    normalized_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized LayerNorm kernel"""
    # Each program handles one element in the batch
    pid = tl.program_id(0)
    
    # Calculate output pointer offset
    output_offset = pid * normalized_size
    
    # Load weight and bias (these are the same for all elements in the dimension)
    weight = tl.load(weight_ptr + tl.arange(0, normalized_size), mask=tl.arange(0, normalized_size) < normalized_size)
    bias = tl.load(bias_ptr + tl.arange(0, normalized_size), mask=tl.arange(0, normalized_size) < normalized_size)
    
    # Load input data for this batch element
    input_data = tl.load(input_ptr + output_offset + tl.arange(0, normalized_size), 
                        mask=tl.arange(0, normalized_size) < normalized_size)
    
    # Compute mean
    sum_val = tl.sum(input_data)
    mean = sum_val / normalized_size
    
    # Compute variance
    input_centered = input_data - mean
    input_centered_sq = input_centered * input_centered
    sum_sq = tl.sum(input_centered_sq)
    var = sum_sq / normalized_size
    
    # Apply normalization
    inv_std = 1.0 / tl.sqrt(var + eps)
    normalized = input_centered * inv_std
    
    # Apply weight and bias
    output = normalized * weight + bias
    
    # Store output
    tl.store(output_ptr + output_offset + tl.arange(0, normalized_size), 
             output, mask=tl.arange(0, normalized_size) < normalized_size)

@torch.fx.wrap  
def optimized_layernorm(input, normalized_shape, weight=None, bias=None, eps=1e-12):
    """Optimized LayerNorm implementation"""
    # Handle different input shapes
    if len(input.shape) == 1:
        # If 1D input, treat as single batch element
        n_elements = 1
        normalized_size = input.shape[0]
    elif len(input.shape) == 2:
        # If 2D input [batch, features], normalize across features
        n_elements = input.shape[0] 
        normalized_size = input.shape[1]
    else:
        raise ValueError(f"Unsupported input shape: {input.shape}")
    
    # Ensure weight and bias exist
    if weight is None:
        weight = torch.ones(normalized_size, dtype=input.dtype, device=input.device)
    if bias is None:
        bias = torch.zeros(normalized_size, dtype=input.dtype, device=input.device)
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Choose optimal block size
    BLOCK_SIZE = 1  # Each program handles one batch element
    
    # Calculate grid size
    grid = (n_elements,)
    
    # Launch kernel
    optimized_layernorm_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=input.numel(),
        normalized_size=normalized_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the optimized function"""
    return optimized_layernorm