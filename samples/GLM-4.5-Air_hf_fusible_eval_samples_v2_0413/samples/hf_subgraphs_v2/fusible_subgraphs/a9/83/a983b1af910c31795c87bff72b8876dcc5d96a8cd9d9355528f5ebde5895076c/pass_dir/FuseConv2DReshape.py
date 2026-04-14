import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor):
    """Pattern: Conv2D + multiply by 1.0 + reshape - fusion opportunity"""
    conv2d = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d * 1.0  # This is a no-op that can be eliminated
    tmp_4 = tmp_3.reshape(-1, 17, 4096)
    return tmp_4

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

# Triton kernel for fused conv2d + reshape (eliminating the no-op multiplication)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 17, 'BLOCK_SIZE_N': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 17, 'BLOCK_SIZE_N': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 17, 'BLOCK_SIZE_N': 2048}, num_warps=8),
    ],
    key=['B', 'C_in', 'H', 'W'],
)
@triton.jit
def fused_conv2d_reshape_kernel(
    input_ptr,      # [B, C_in, H, W]
    weight_ptr,     # [C_out, C_in, 1, 1]
    bias_ptr,       # [C_out]
    output_ptr,     # [B, 17, 4096] (reshaped output)
    B, C_in, H, W, 
    BLOCK_SIZE_M: tl.constexpr,  # M dimension (17) block size  
    BLOCK_SIZE_N: tl.constexpr,  # N dimension (B*H*W) block size
):
    # Each program handles a block of 17 x (B*H*W)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate ranges for this program
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, 17)  # Only 17 output channels
    n_start = pid_n * BLOCK_SIZE_N
    n_end = min((pid_n + 1) * BLOCK_SIZE_N, B * H * W)
    
    # Calculate input indices
    batch_idx = n_start // (H * W)
    spatial_idx = n_start % (H * W)
    h_idx = spatial_idx // W
    w_idx = spatial_idx % W
    
    # Load bias for this output channel block (only first 17 channels)
    if m_start < 17:
        bias_value = tl.load(bias_ptr + m_start).to(tl.float32)
    else:
        bias_value = tl.cast(0.0, tl.float32)
    
    # Process this block
    acc = bias_value
    
    # Loop over input channels to compute dot product (1x1 convolution)
    for k in range(C_in):
        # Load weight value [C_out, C_in, 1, 1] - only first 17 channels
        weight_val = tl.load(weight_ptr + m_start * C_in * 1 * 1 + k * 1 * 1 + 0 * 1 + 0)
        
        # Load input value [B, C_in, H, W] at (batch_idx, k, h_idx, w_idx)
        input_val = tl.load(input_ptr + batch_idx * C_in * H * W + k * H * W + h_idx * W + w_idx)
        
        # Accumulate dot product
        acc += weight_val * input_val
    
    # Store result directly in reshaped format [B, 17, 4096]
    # For spatial position and batch, convert to output format
    if m_start < 17 and n_start < B * H * W:
        # Store in output tensor: output[batch_idx, m_start, spatial_idx]
        tl.store(output_ptr + batch_idx * 17 * 4096 + m_start * 4096 + spatial_idx, acc)

@torch.fx.wrap
def fused_conv2d_reshape(input_tensor, weight_tensor, bias_tensor):
    """Fused conv2d + reshape with no-op elimination"""
    B, C_in, H, W = input_tensor.shape
    C_out = weight_tensor.shape[0]  # Should be 17
    
    # Calculate final output shape: [-1, 17, 4096]
    # Since H*W = 64*64 = 4096, this becomes [B, 17, 4096]
    output_shape = (B, 17, 4096)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Setup Triton kernel with autotune
    # Calculate grid dimensions
    grid_m = (17 + 17 - 1) // 17  # Should always be 1 (for BLOCK_SIZE_M = 17)
    grid_n = (B * H * W + 2048 - 1) // 2048  # Conservative estimate for largest BLOCK_SIZE_N
    
    # Launch kernel with autotune - Triton will handle optimal block sizes
    fused_conv2d_reshape_kernel[grid_m, grid_n](
        input_tensor,
        weight_tensor,
        bias_tensor,
        output,
        B, C_in, H, W
    )
    
    return output

def replacement_func():
    return fused_conv2d_reshape