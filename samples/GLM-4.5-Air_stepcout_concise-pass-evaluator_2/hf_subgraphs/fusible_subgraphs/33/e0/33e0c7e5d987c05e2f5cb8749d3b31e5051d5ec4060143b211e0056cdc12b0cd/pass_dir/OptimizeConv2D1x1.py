import torch
import triton
import triton.language as tl


def pattern(x, weight, bias):
    return torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)


def replacement_args(x, weight, bias):
    return x, weight, bias


@triton.jit
def conv2d_1x1_kernel(
    x_ptr,        # Input tensor [N, C_in, H, W] 
    weight_ptr,   # Weight [C_out, C_in, 1, 1]
    bias_ptr,     # Bias [C_out]
    out_ptr,      # Output [N, C_out, H, W]
    n,            # Batch size N
    c_in,         # Input channels C_in
    c_out,        # Output channels C_out  
    h, w,         # Spatial dimensions H, W
    BLOCK_SIZE: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Optimized 1x1 convolution kernel"""
    # Each program handles one output channel and a block of spatial data
    pid_c = tl.program_id(0)  # Output channel
    pid_h = tl.program_id(1)  # Height block
    pid_w = tl.program_id(2)  # Width block
    
    # Check bounds for output channel
    if pid_c >= c_out:
        return
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + pid_c)
    
    # Process spatial block
    block_h_start = pid_h * BLOCK_SIZE
    block_w_start = pid_w * BLOCK_COLS
    
    # Calculate spatial offsets within this block
    h_offsets = block_h_start + tl.arange(0, BLOCK_SIZE)
    w_offsets = block_w_start + tl.arange(0, BLOCK_COLS)
    
    # Create masks for spatial dimensions
    h_mask = h_offsets < h
    w_mask = w_offsets < w
    
    # Process batch dimension
    for b in range(n):
        # Load weights for this output channel (all input channels)
        weight_values = tl.load(weight_ptr + pid_c * c_in + tl.arange(0, c_in), 
                              other=0.0)
        
        # Initialize output accumulator 
        result = tl.zeros((BLOCK_SIZE, BLOCK_COLS), dtype=tl.float32)
        
        # Loop through input channels (reduce dimension)
        for c_idx in range(c_in):
            # Load input slice for this input channel
            x_ptr_base = x_ptr + (b * c_in + c_idx) * h * w
            x_values = tl.load(x_ptr_base + h_offsets[:, None] * w + w_offsets[None, :], 
                             mask=h_mask[:, None] & w_mask[None, :], other=0.0)
            
            # Multiply and accumulate
            result += x_values * weight_values[c_idx]
        
        # Add bias and store result
        result += bias
        
        # Store output for this batch position
        out_ptr_base = out_ptr + (b * c_out + pid_c) * h * w
        tl.store(out_ptr_base + h_offsets[:, None] * w + w_offsets[None, :], 
                result, mask=h_mask[:, None] & w_mask[None, :])


@torch.fx.wrap
def optimized_conv2d_1x1(x, weight, bias):
    """Optimized 1x1 convolution using Triton"""
    x_shape = x.shape
    weight_shape = weight.shape
    bias_shape = bias.shape
    
    # Validate shapes
    assert len(x_shape) == 4, f"Expected 4D input, got {len(x_shape)}D"
    assert len(weight_shape) == 4, f"Expected 4D weight, got {len(weight_shape)}D"
    assert weight_shape[2] == 1 and weight_shape[3] == 1, "Expected 1x1 convolution"
    assert x_shape[1] == weight_shape[1], f"Channel mismatch: {x_shape[1]} vs {weight_shape[1]}"
    
    # Extract dimensions
    n, c_in, h, w = x_shape
    c_out = weight_shape[0]
    
    # Validate bias
    assert bias_shape[0] == c_out, f"Bias shape mismatch: expected {c_out}, got {bias_shape[0]}"
    
    # Create output tensor
    out = torch.empty((n, c_out, h, w), dtype=x.dtype, device=x.device)
    
    # Calculate grid dimensions
    BLOCK_SIZE = 16  # Height block size
    BLOCK_COLS = 32  # Width block size
    
    # Calculate number of programs needed
    num_h_blocks = (h + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_w_blocks = (w + BLOCK_COLS - 1) // BLOCK_COLS
    
    # Launch kernel
    conv2d_1x1_kernel[(c_out, num_h_blocks, num_w_blocks)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n=n,
        c_in=c_in,
        c_out=c_out,
        h=h,
        w=w,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_COLS=BLOCK_COLS,
    )
    
    return out


def replacement_func():
    return optimized_conv2d_1x1