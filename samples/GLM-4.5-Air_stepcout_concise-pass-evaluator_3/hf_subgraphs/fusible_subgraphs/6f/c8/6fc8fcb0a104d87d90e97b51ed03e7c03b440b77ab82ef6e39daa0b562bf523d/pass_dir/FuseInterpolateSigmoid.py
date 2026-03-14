import torch
import triton
import triton.language as tl

@triton.jit
def interpolate_sigmoid_kernel(
    input_ptr,
    output_ptr,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """
    Triton kernel that fuses bilinear interpolation and sigmoid activation.
    Efficiently upsamples from input size [N, C, H_in, W_in] to [N, C, H_out, W_out]
    and applies sigmoid in a single kernel.
    """
    # Calculate grid positions
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output coordinates
    h_out = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_out = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    h_out = h_out < H_out
    w_out = w_out < W_out
    
    # Calculate interpolation coordinates for output position
    # For bilinear interpolation, calculate corresponding input coordinates
    scale_h = H_in / H_out
    scale_w = W_in / W_out
    
    # Calculate the input coordinates for each output pixel
    h_in_float = h_out * scale_h
    w_in_float = w_out * scale_w
    
    # Get integer parts and fractional parts for interpolation
    h0 = tl.floor(h_in_float).to(tl.int32)
    w0 = tl.floor(w_in_float).to(tl.int32)
    h1 = h0 + 1
    w1 = w0 + 1
    
    # Calculate fractional weights
    h_frac = h_in_float - h0.to(tl.float32)
    w_frac = w_in_float - w0.to(tl.float32)
    
    # Clamp coordinates to valid range
    h0 = tl.where(h0 < H_in, h0, H_in - 1)
    w0 = tl.where(w0 < W_in, w0, W_in - 1)
    h1 = tl.where(h1 < H_in, h1, H_in - 1)
    w1 = tl.where(w1 < W_in, w1, W_in - 1)
    
    # Calculate pointers for bilinear interpolation
    # Input stride: [H_in * W_in * C, W_in * C, C, 1]
    input_base = pid_n * H_in * W_in * C + pid_c * H_in * W_in
    
    # Load the four neighboring pixels for bilinear interpolation
    # We need to load values at (h0, w0), (h0, w1), (h1, w0), (h1, w1)
    idx00 = input_base + h0 * W_in + w0
    idx01 = input_base + h0 * W_in + w1
    idx10 = input_base + h1 * W_in + w0
    idx11 = input_base + h1 * W_in + w1
    
    # Load with bounds checking
    val00 = tl.load(input_ptr + idx00, mask=h_out & w_out, other=0.0)
    val01 = tl.load(input_ptr + idx01, mask=h_out & w_out, other=0.0)
    val10 = tl.load(input_ptr + idx10, mask=h_out & w_out, other=0.0)
    val11 = tl.load(input_ptr + idx11, mask=h_out & w_out, other=0.0)
    
    # Perform bilinear interpolation
    # Interpolate horizontally first
    val0 = val00 * (1 - w_frac) + val01 * w_frac
    val1 = val10 * (1 - w_frac) + val11 * w_frac
    
    # Then interpolate vertically
    interpolated = val0 * (1 - h_frac) + val1 * h_frac
    
    # Apply sigmoid activation
    sigmoid_result = 1.0 / (1.0 + tl.exp(-interpolated))
    
    # Calculate output pointer and store result
    output_idx = pid_n * H_out * W_out * C + pid_c * H_out * W_out + pid_h * BLOCK_SIZE_H * W_out + pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_H)[:, None] * W_out + tl.arange(0, BLOCK_SIZE_W)[None, :]
    
    # Reshape for proper storage
    sigmoid_result = sigmoid_result.reshape([-1])
    mask = (tl.arange(0, BLOCK_SIZE_H * BLOCK_SIZE_W) < (BLOCK_SIZE_H * BLOCK_SIZE_W)) & h_out.reshape([-1])[:, None] & w_out.reshape([-1])[None, :]
    
    # Store the result
    tl.store(output_ptr + output_idx, sigmoid_result, mask=mask.reshape([-1]))

@torch.fx.wrap
def fuse_interpolate_sigmoid(input_tensor, output_size=(640, 640)):
    """
    Function that fuses bilinear interpolation and sigmoid activation.
    Efficiently upsamples and applies sigmoid in a single operation.
    """
    N, C, H_in, W_in = input_tensor.shape
    H_out, W_out = output_size
    
    # Calculate optimal block sizes
    BLOCK_SIZE_W = 32  # Optimal for memory coalescing
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_C = 64  # Process multiple channels together
    BLOCK_SIZE_N = 1   # Process one batch at a time for simplicity
    
    # Calculate grid dimensions
    grid_h = (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Create output tensor
    output = torch.empty((N, C, H_out, W_out), 
                        dtype=input_tensor.dtype, 
                        device=input_tensor.device)
    
    # Launch the kernel
    if N > 0 and C > 0:
        interpolate_sigmoid_kernel[(N, grid_c, grid_h, grid_w)](
            input=input_tensor,
            output=output,
            N=N,
            C=C,
            H_in=H_in,
            W_in=W_in,
            H_out=H_out,
            W_out=W_out,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
        )
    
    return output

# Pattern matching function - matches interpolate followed by sigmoid
def pattern(input_tensor):
    """Matches interpolate operation followed by sigmoid activation"""
    tmp_3 = torch.nn.functional.interpolate(input_tensor, size=(640, 640), mode='bilinear')
    tmp_9 = torch.nn.functional.sigmoid(tmp_3)
    return tmp_9  # Return only the final result as it's the observable output

# Argument extraction function
def replacement_args(input_tensor):
    """Extract arguments needed for the fused interpolate+sigmoid operation"""
    return (input_tensor,)

# Replacement function - returns the fused kernel wrapper
def replacement_func():
    return fuse_interpolate_sigmoid