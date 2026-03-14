import torch
import triton
import triton.language as tl

def pattern(inv_freq, temp_4):
    """
    Pattern matching for fused cos/sin computation in rotary embeddings.
    Matches the sequence:
    - tmp_6 = tmp_5.cos()
    - tmp_7 = tmp_6[None, None, slice(None, None, None), slice(None, None, None)]
    - tmp_8 = tmp_5.sin()  
    - tmp_9 = tmp_8[None, None, slice(None, None, None), slice(None, None, None)]
    
    Returns the expanded cos and sin tensors that are observable outputs
    """
    cos_tensor = temp_4.cos()
    sin_tensor = temp_4.sin()
    
    # Add dimensions with slicing to match the pattern
    expanded_cos = cos_tensor[None, None, slice(None, None, None), slice(None, None, None)]
    expanded_sin = sin_tensor[None, None, slice(None, None, None), slice(None, None, None)]
    
    return expanded_cos, expanded_sin

def replacement_args(inv_freq, temp_4):
    """Extract arguments for the fused kernel - just the input tensor and position info"""
    return inv_freq, temp_4

@triton.jit
def fused_cos_sin_kernel(
    input_ptr,
    cos_out_ptr,
    sin_out_ptr,
    n_cols,
    add_dims: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused cos/sin computation kernel optimized for rotary embeddings.
    Computes both cos and sin in a single kernel launch.
    """
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute both cos and sin in single operation
    cos_val = tl.cos(x)
    sin_val = tl.sin(x)
    
    # Store results
    tl.store(cos_out_ptr + offsets, cos_val, mask=mask)
    tl.store(sin_out_ptr + offsets, sin_val, mask=mask)

@torch.fx.wrap  
def fused_cos_sin_kernel_wrapper(input_tensor):
    """Wrapper function to launch the fused cos/sin kernel"""
    # Get input tensor properties
    n_cols = input_tensor.shape[-1]
    
    # Create output tensors
    cos_out = torch.empty_like(input_tensor)
    sin_out = torch.empty_like(input_tensor)
    
    # Set block size and launch grid
    BLOCK_SIZE = 1024
    num_programs = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the fused kernel
    fused_cos_sin_kernel[(num_programs,)](
        input_ptr=input_tensor,
        cos_out_ptr=cos_out,
        sin_out_ptr=sin_out,
        n_cols=n_cols,
        add_dims=True,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Add the extra dimensions to match the original pattern
    expanded_cos = cos_out[None, None, slice(None, None, None), slice(None, None, None)]
    expanded_sin = sin_out[None, None, slice(None, None, None), slice(None, None, None)]
    
    return expanded_cos, expanded_sin

def replacement_func():
    """Return the optimized fused kernel function"""
    return fused_cos_sin_kernel_wrapper