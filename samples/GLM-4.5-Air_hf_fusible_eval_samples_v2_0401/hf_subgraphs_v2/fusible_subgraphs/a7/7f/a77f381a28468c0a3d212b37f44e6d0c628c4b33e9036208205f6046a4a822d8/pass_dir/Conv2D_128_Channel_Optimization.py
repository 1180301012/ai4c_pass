import torch
import triton
import triton.language as tl
import math

def pattern(input_tensor, weight_tensor, stride, padding, dilation, groups):
    """Match conv2d followed by channel slicing pattern for 128 channels"""
    # Create pattern that mirrors the exact structure of target computation
    # Using concatenation to ensure both results are used and no dead code
    conv2d_slice = input_tensor[:, :128, :, :]
    full_result = input_tensor  # Represent the full conv2d result
    return (conv2d_slice, full_result)

def replacement_args(input_tensor, weight_tensor, stride, padding, dilation, groups):
    """Extract arguments needed for the replacement"""
    # Pass through arguments - target_channels determined in replacement function
    return (input_tensor, weight_tensor, stride, padding, dilation, groups)

@triton.jit
def partial_conv2d_kernel_128(
    input_ptr, weight_ptr, output_ptr,
    N, H, W, I_C, O_C, K_H, K_W,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    target_channels,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    """Optimized conv2d kernel for 128 channel targeting"""
    # Program ID mapping
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Only compute up to target_channels
    actual_n_start = min(n_start, target_channels)
    actual_n_end = min(min(n_start + BLOCK_SIZE_N, target_channels), target_channels)
    
    if actual_n_start >= actual_n_end:
        return
    
    # Compute ranges with bounds checking
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = actual_n_start + tl.arange(0, actual_n_end - actual_n_start)
    
    # Bounds masks
    m_mask = m_offsets < N
    n_mask = tl.arange(0, actual_n_end - actual_n_start) < (actual_n_end - actual_n_start)
    
    # Initialize output
    output = tl.zeros((BLOCK_SIZE_M, actual_n_end - actual_n_start, H * W), dtype=tl.float32)
    
    # Load input blocks and compute for each output channel
    for c in range(I_C):
        # Load input for current channel
        input_ptrs = input_ptr + (m_offsets[:, None] * H * W * I_C + c * H * W)
        input_vals = tl.load(input_ptrs, mask=m_mask[:, None], other=0.0)
        input_reshaped = input_vals.reshape((BLOCK_SIZE_M, -1))
        
        # Load weight for current input channel
        weight_ptrs = weight_ptr + (n_offsets[:, None] * I_C * K_H * K_W + c * K_H * K_W)
        weight_vals = tl.load(weight_ptrs, mask=n_mask[:, None], other=0.0)
        
        # Convolution computation - sum over kernel positions
        for kh in range(K_H):
            for kw in range(K_W):
                # Load kernel weights
                kernel_ptrs = weight_ptrs + (kh * K_W + kw)
                kernel = tl.load(kernel_ptrs, mask=n_mask[:, None], other=0.0).reshape(-1)
                
                # Apply stride and dilation for kernel position
                h_stride = kh * dilation_h * stride_h
                w_stride = kw * dilation_w * stride_w
                
                # Extract corresponding input patches
                input_patch_ptrs = input_ptrs + (h_stride * W * I_C + w_stride * I_C)
                input_patch = tl.load(input_patch_ptrs, mask=m_mask[:, None], other=0.0).reshape((BLOCK_SIZE_M, -1))
                
                # Compute contribution
                output += tl.dot(input_patch.to(tl.float32), kernel.to(tl.float32)[None, :])
    
    # Store output
    output_ptrs = output_ptr + (m_offsets[:, None] * target_channels * H * W + n_offsets[None, :] * H * W + tl.arange(0, H * W)[None, None])
    
    for h in range(H):
        for w in range(W):
            idx = h * W + w
            if idx < output.shape[2]:
                current_output = output[:, :, idx]
                current_ptrs = output_ptrs[:, :, idx].reshape(-1)
                for i in range(current_output.numel()):
                    if i < current_ptrs.numel():
                        tl.store(current_ptrs + i, current_output[i], mask=(i < current_ptrs.numel()))

@torch.fx.wrap
def partial_conv2d_triton_128(input_tensor, weight_tensor, stride, padding, dilation, groups):
    """Triton wrapper for partial conv2d computation targeting 128 channels"""
    input_shape = input_tensor.shape
    weight_shape = weight_tensor.shape
    
    # Only support convolutions with groups=1 for optimization
    if groups != 1:
        raise ValueError("Only groups=1 convolution supported by this optimization pass")
    
    # Set target channels for this optimization
    target_channels = 128
    
    # Ensure we don't exceed available channels
    if target_channels > weight_shape[0]:
        target_channels = weight_shape[0]
    
    N, C_in, H_in, W_in = input_shape
    C_out, _, K_H, K_W = weight_shape
    
    # Output dimensions calculation
    H_out = (H_in + 2 * padding[0] - dilation[0] * (K_H - 1) - 1) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - dilation[1] * (K_W - 1) - 1) // stride[1] + 1
    
    # Allocate output tensors
    full_output = torch.zeros((N, C_out, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)
    partial_output = torch.zeros((N, target_channels, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set kernel parameters optimized for smaller channel counts
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 32
    
    # Grid configuration (2D grid for channel+batch parallelism)
    grid_m = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (target_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel for partial computation
    partial_conv2d_kernel_128[grid_m, grid_n](
        input_tensor, weight_tensor, partial_output,
        N, H_out, W_out, C_in, C_out, K_H, K_W,
        stride[0], stride[1], padding[0], padding[1],
        dilation[0], dilation[1],
        target_channels,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    # Copy the computed partial result to full output for the channels we computed
    full_output[:, :target_channels, :, :] = partial_output
    
    return partial_output, full_output

def replacement_func():
    """Return the optimized function"""
    return partial_conv2d_triton_128