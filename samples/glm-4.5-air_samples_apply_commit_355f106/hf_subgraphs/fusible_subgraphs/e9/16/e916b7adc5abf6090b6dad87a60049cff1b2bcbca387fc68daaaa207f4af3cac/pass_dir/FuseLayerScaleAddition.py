import torch
import triton
import triton.language as tl

def pattern(conv_out, gamma, residual_input):
    """
    Pattern: Dropout * Layer_Scale + Addition
    This captures: dropout(conv_out, 0.0) * gamma + residual_input
    Since dropout with p=0.0 is identity, this simplifies to conv_out * gamma + residual_input
    """
    # dropout with p=0.0 is identity operation
    layer_scaled = conv_out * gamma
    result = residual_input + layer_scaled
    return result

def replacement_args(conv_out, gamma, residual_input):
    return (conv_out, gamma, residual_input)

@triton.jit
def fused_layer_scale_add_kernel(
    conv_out_ptr,
    gamma_ptr,
    residual_input_ptr,
    out_ptr,
    n_elements,
    n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    conv_out = tl.load(conv_out_ptr + offsets, mask=mask, other=0.0)
    residual_input = tl.load(residual_input_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate channel index for gamma broadcasting
    # Since gamma is [C, 1, 1], we need to extract the correct channel value
    channel_idx = (offsets % n_channels)  # Map each position to its channel
    
    # Load gamma with proper bounds checking
    gamma = tl.load(gamma_ptr + channel_idx, mask=channel_idx < n_channels, other=1.0)
    
    # Fused operation: conv_out * gamma + residual_input
    # This eliminates the intermediate dropout operation and fuses multiply-add
    layer_scaled = conv_out * gamma
    result = residual_input + layer_scaled
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)
    
@torch.fx.wrap
def fused_layer_scale_addition(conv_out, gamma, residual_input):
    # Get tensor shape
    B, C, H, W = conv_out.shape
    n_elements = B * C * H * W
    n_channels = C  # Number of channels for gamma broadcasting
    
    # Create output tensor
    out = torch.empty_like(conv_out)
    
    # Use optimal block size based on tensor size for better performance
    if n_elements > 1000000:
        BLOCK_SIZE = 1024
    elif n_elements > 100000:
        BLOCK_SIZE = 512
    elif n_elements > 10000:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 128
        
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch fused layer scale and addition kernel
    fused_layer_scale_add_kernel[(num_programs,)](
        conv_out_ptr=conv_out,
        gamma_ptr=gamma,
        residual_input_ptr=residual_input,
        out_ptr=out,
        n_elements=n_elements,
        n_channels=n_channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_layer_scale_addition