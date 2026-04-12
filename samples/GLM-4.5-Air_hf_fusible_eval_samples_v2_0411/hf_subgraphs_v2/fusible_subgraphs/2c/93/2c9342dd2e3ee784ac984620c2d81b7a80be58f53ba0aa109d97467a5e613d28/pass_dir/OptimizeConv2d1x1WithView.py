import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    """Optimize 1x1 Conv2D + View operation fusion - exact pattern matching"""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    conv2d_shape = conv2d.shape
    result = conv2d.view(conv2d_shape[0], 1, -1)
    return result

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def optimized_conv1x1_kernel(
    input_ptr,      # [N, C_in, H, W]
    weight_ptr,     # [C_out, C_in, 1, 1]
    bias_ptr,       # [C_out]
    output_ptr,     # [N, 1, C_out*H*W]
    N: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized 1x1 convolution that directly outputs in view format"""
    pid = tl.program_id(0)
    
    # Each program handles one output position in the flattened output
    # Output is [N, 1, C_out*H*W], so we process one element at a time
    linear_idx = pid
    
    if linear_idx >= N * C_out * H * W:
        return
    
    # Convert linear index to output coordinates
    out_c = linear_idx // (H * W)
    spatial_idx = linear_idx % (H * W)
    spatial_h = spatial_idx // W
    spatial_w = spatial_idx % W
    n_idx = out_c // (C_out * H * W // N)  # Simplified for single batch processing
    
    # Perform 1x1 convolution: sum over input channels
    acc = 0.0
    for c_in in range(C_in):
        # Load input
        input_idx = ((n_idx * C_in + c_in) * H + spatial_h) * W + spatial_w
        val_in = tl.load(input_ptr + input_idx, mask=(n_idx < N) & (c_in < C_in) & (spatial_h < H) & (spatial_w < W), other=0.0)
        
        # Load weight and bias
        weight_idx = out_c * C_in + c_in
        val_weight = tl.load(weight_ptr + weight_idx, mask=(out_c < C_out) & (c_in < C_in), other=0.0)
        val_bias = tl.load(bias_ptr + out_c, mask=(out_c < C_out), other=0.0)
        
        acc += val_in * val_weight
    
    acc += tl.load(bias_ptr + out_c, mask=(out_c < C_out), other=0.0)
    
    # Store result in the flattened output format [N, 1, C_out*H*W]
    output_idx = n_idx * (C_out * H * W) + linear_idx
    tl.store(output_ptr + output_idx, acc, mask=linear_idx < N * C_out * H * W)

@torch.fx.wrap
def optimized_conv1x1_view(in_2, in_1, in_0):
    """Optimized 1x1 convolution with direct view output"""
    N, C_in, H, W = in_2.shape
    C_out = in_1.shape[0]
    
    # Output should be [N, 1, C_out*H*W]
    output_shape = (N, 1, C_out * H * W)
    
    # For this specific case where output has middle dimension 1, we can optimize
    # Since we're flattening to [N, 1, -1], we can directly compute in that format
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    total_elements = N * C_out * H * W
    BLOCK_SIZE = 256  # Adjust based on GPU architecture
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_conv1x1_kernel[grid_size](
        in_2,
        in_1,
        in_0,
        output,
        N, C_in, C_out, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_conv1x1_view