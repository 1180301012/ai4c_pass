import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor):
    """Pattern: 1x1 Conv2D which can be optimized as matrix multiplication"""
    conv2d = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)
    return conv2d

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

# Triton kernel for optimized 1x1 conv2d as matrix multiplication
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 1024}, num_warps=4),
    ],
    key=['B', 'C_in', 'H', 'W', 'C_out'],
)
@triton.jit
def optimized_1x1_conv2d_kernel(
    input_ptr,      # [B, C_in, H, W]
    weight_ptr,     # [C_out, C_in, 1, 1]
    bias_ptr,       # [C_out]
    output_ptr,     # [B, C_out, H, W]
    B, C_in, H, W, C_out,
    BLOCK_SIZE_M: tl.constexpr,  # M dimension (C_out) block size
    BLOCK_SIZE_N: tl.constexpr,  # N dimension (B*H*W) block size
):
    # Each program handles a block of C_out x (B*H*W)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges for this program
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, C_out)
    n_start = pid_n * BLOCK_SIZE_N
    n_end = min((pid_n + 1) * BLOCK_SIZE_N, B * H * W)
    
    # Calculate input indices
    batch_idx = n_start // (H * W)
    spatial_idx = n_start % (H * W)
    h_idx = spatial_idx // W
    w_idx = spatial_idx % W
    
    # Compute output pointer offset for conv2d output format
    output_offset = batch_idx * C_out * H * W + m_start * H * W + h_idx * W + w_idx
    
    # Load bias for this output channel block
    if m_start < C_out:
        bias_value = tl.load(bias_ptr + m_start).to(tl.float32)
    else:
        bias_value = tl.cast(0.0, tl.float32)
    
    # Process this block
    acc = bias_value
    
    # Loop over input channels to compute dot product (1x1 convolution)
    for k in range(C_in):
        # Load weight value [C_out, C_in, 1, 1]
        weight_val = tl.load(weight_ptr + m_start * C_in * 1 * 1 + k * 1 * 1 + 0 * 1 + 0)
        
        # Load input value [B, C_in, H, W] at (batch_idx, k, h_idx, w_idx)
        input_val = tl.load(input_ptr + batch_idx * C_in * H * W + k * H * W + h_idx * W + w_idx)
        
        # Accumulate dot product
        acc += weight_val * input_val
    
    # Store result in conv2d output format
    if m_start < C_out and n_start < B * H * W:
        tl.store(output_ptr + output_offset, acc)

@torch.fx.wrap
def optimized_1x1_conv2d(input_tensor, weight_tensor, bias_tensor):
    """Optimized 1x1 conv2d using matrix multiplication approach"""
    B, C_in, H, W = input_tensor.shape
    C_out = weight_tensor.shape[0]
    
    # Compute output shape
    output_shape = (B, C_out, H, W)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Setup Triton kernel with autotune
    # Calculate grid dimensions
    grid_m = (C_out + 128 - 1) // 128  # Conservative estimate for largest BLOCK_SIZE_M
    grid_n = (B * H * W + 2048 - 1) // 2048  # Conservative estimate for largest BLOCK_SIZE_N
    
    # Launch kernel with autotune - Triton will handle optimal block sizes
    optimized_1x1_conv2d_kernel[grid_m, grid_n](
        input_tensor,
        weight_tensor,
        bias_tensor,
        output,
        B, C_in, H, W, C_out
    )
    
    return output

def replacement_func():
    return optimized_1x1_conv2d