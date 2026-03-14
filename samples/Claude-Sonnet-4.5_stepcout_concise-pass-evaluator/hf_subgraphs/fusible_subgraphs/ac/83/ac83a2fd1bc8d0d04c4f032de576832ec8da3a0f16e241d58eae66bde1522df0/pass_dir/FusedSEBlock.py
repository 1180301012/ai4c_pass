import torch
import triton
import triton.language as tl

# Pattern matching function - matches sigmoid -> multiply -> hardtanh
def pattern(conv_out, input_tensor):
    """ 
    Match the fused pattern after conv2d:
    sigmoid -> multiply -> hardtanh
    """
    tmp_3 = conv_out.sigmoid()
    tmp_4 = input_tensor * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5

# Argument extraction function
def replacement_args(conv_out, input_tensor):
    return (conv_out, input_tensor)

# Optimized fused kernel for sigmoid + multiply + hardtanh 
@triton.jit
def fused_se_kernel(
    conv_out_ptr,
    input_ptr,
    output_ptr,
    n_elements,
    channels,
    hw,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: sigmoid(conv_out) * input, then hardtanh
    Optimized for memory coalescing and reduced index computation
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which elements this program handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate batch and channel indices for broadcast
    # Optimize: compute once, reuse
    temp = offsets // hw
    c_idx = temp % channels
    b_idx = temp // channels
    
    # Load input values (coalesced)
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load conv output (broadcast from [B, C, 1, 1])
    conv_offsets = b_idx * channels + c_idx
    conv_vals = tl.load(conv_out_ptr + conv_offsets, mask=mask, other=0.0)
    
    # Fused compute: sigmoid, multiply, and clamp
    sigmoid_vals = tl.sigmoid(conv_vals)
    result = input_vals * sigmoid_vals
    
    # Apply hardtanh (clamp between 0.0 and 6.0)
    result = tl.minimum(tl.maximum(result, 0.0), 6.0)
    
    # Store result (coalesced)
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_se_forward(conv_out, input_tensor):
    """
    Wrapper function that launches fused kernel for sigmoid + multiply + hardtanh
    """
    # Get dimensions
    batch_size, channels, height, width = input_tensor.shape
    hw = height * width
    n_elements = batch_size * channels * hw
    
    # Allocate output
    output = torch.empty_like(input_tensor)
    
    # Use fixed BLOCK_SIZE for less overhead
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_se_kernel[grid](
        conv_out,
        input_tensor,
        output,
        n_elements,
        channels,
        hw,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (no arguments, returns function reference)
def replacement_func():
    return fused_se_forward