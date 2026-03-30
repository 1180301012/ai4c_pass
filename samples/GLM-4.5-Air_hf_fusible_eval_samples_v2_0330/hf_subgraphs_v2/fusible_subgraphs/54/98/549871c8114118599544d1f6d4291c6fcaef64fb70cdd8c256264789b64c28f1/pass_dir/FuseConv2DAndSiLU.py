import torch
import triton
import triton.language as tl

def pattern(conv2d):
    """Pattern to match SiLU operation after conv2d"""
    # This matches the SiLU operation that uses the result of conv2d
    tmp_3 = torch.nn.functional.silu(conv2d, inplace=False)
    return tmp_3

def replacement_args(conv2d, in_1, in_0):
    """Extract arguments for the replacement kernel"""
    # The replacement needs the conv2d result and the original conv weights/bias
    return (conv2d, in_1, in_0)

@triton.jit
def fused_conv2d_silu_kernel(
    conv2d_ptr,
    weight_ptr,
    bias_ptr, 
    out_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """Fused Conv2D + SiLU kernel"""
    # Each program handles one output channel
    m = tl.program_id(0)
    
    # Load bias for this output channel
    bias_val = tl.load(bias_ptr + m)
    
    # Process each spatial position
    for hi in range(height):
        for wi in range(width):
            # Initialize accumulator with bias
            acc = bias_val
            
            # Compute dot product between input channels and weights
            for k in range(0, in_channels, BLOCK_SIZE_K):
                # Load weights for this output channel and input channels
                weight_idx = m * in_channels + k
                weight_end = min(k + BLOCK_SIZE_K, in_channels)
                mask_k = range(weight_end - k)
                weights = tl.load(weight_ptr + weight_idx, mask=mask_k)
                
                # Load corresponding input channels for this spatial position
                # Note: for 1x1 conv, spatial dimensions don't change the channel index
                input_idx = k  # We're processing all spatial positions together
                input_end = min(k + BLOCK_SIZE_K, in_channels)
                mask_input = range(input_end - k)
                inputs = tl.load(conv2d_ptr + input_idx, mask=mask_input)
                
                # Accumulate dot product
                for i in range(len(weights)):
                    acc += weights[i] * inputs[i]
            
            # Apply SiLU activation: x * sigmoid(x)
            x = acc
            sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
            silu_out = x * sigmoid_x
            
            # Store result for all spatial positions
            for hi in range(height):
                for wi in range(width):
                    output_idx = m * height * width + hi * width + wi
                    tl.store(out_ptr + output_idx, silu_out)

@torch.fx.wrap  
def fused_conv2d_silu(conv2d, in_1, in_0):
    """Fused Conv2D + SiLU wrapper"""
    batch_size, in_channels, height, width = conv2d.shape
    out_channels, _, _, _ = in_1.shape
    
    # Create output tensor
    out = torch.empty((batch_size, out_channels, height, width), dtype=conv2d.dtype, device=conv2d.device)
    
    # Launch kernel
    BLOCK_SIZE_M = 64  # Number of output channels per program
    BLOCK_SIZE_K = 128  # Vectorize over input channels
    
    num_programs = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    fused_conv2d_silu_kernel[(num_programs,)](
        conv2d,
        in_1,
        in_0,
        out,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K
    )
    
    return out

def replacement_func():
    """Return the fused kernel function"""
    return fused_conv2d_silu