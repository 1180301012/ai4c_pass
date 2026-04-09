import torch
import triton
import triton.language as tl

# Pattern matching function for element-wise multiplication with masking
def pattern(layer_norm_output, expanded_float_mask):
    """
    Match the element-wise multiplication between layer norm output and expanded mask
    Args:
        layer_norm_output: [1, 16, 768], torch.float16/bfloat16
        expanded_float_mask: [1, 16, 768], torch.float32
    Returns:
        result: [1, 16, 768], torch.float16/bfloat16 (same dtype as layer_norm_output)
    """
    result = layer_norm_output * expanded_float_mask
    return result

def replacement_args(layer_norm_output, expanded_float_mask):
    return (layer_norm_output, expanded_float_mask)

@triton.jit
def optimized_multiply_kernel(
    layer_norm_output_ptr,
    mask_ptr,
    output_ptr,
    batch, seq_len, hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized element-wise multiplication kernel
    Handles dtype conversion and broadcasting efficiently
    """
    pid = tl.program_id(0)
    
    # Calculate offsets for this program
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < batch * seq_len * hidden_size
    
    # Load layer norm output with vectorized access
    ln_out = tl.load(layer_norm_output_ptr + offset, mask=mask, other=0.0)
    
    # Load attention mask - Triton will handle type casting automatically
    mask_val = tl.load(mask_ptr + offset, mask=mask, other=0.0)
    
    # Perform multiplication with automatic type casting to match input dtype
    result = ln_out * mask_val
    
    # Store result
    tl.store(output_ptr + offset, result, mask=mask)

@torch.fx.wrap
def optimized_multiply(layer_norm_output, expanded_float_mask):
    """
    Optimized element-wise multiplication function with performance tuning
    Args:
        layer_norm_output: [1, 16, 768], torch.float16/bfloat16
        expanded_float_mask: [1, 16, 768], torch.float32
    Returns:
        result: [1, 16, 768], torch.float16/bfloat16
    """
    batch, seq_len, hidden_size = layer_norm_output.shape
    total_elements = batch * seq_len * hidden_size
    
    # Use optimal block size for GPU (smaller block size for better throughput)
    BLOCK_SIZE = 256  # Smaller block size for better occupancy on small tensors
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype as layer norm output
    output = torch.empty_like(layer_norm_output)
    
    # Launch kernel with optimal configuration
    optimized_multiply_kernel[(num_programs,)](
        layer_norm_output_ptr=layer_norm_output,
        mask_ptr=expanded_float_mask,
        output_ptr=output,
        batch=batch,
        seq_len=seq_len,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_multiply