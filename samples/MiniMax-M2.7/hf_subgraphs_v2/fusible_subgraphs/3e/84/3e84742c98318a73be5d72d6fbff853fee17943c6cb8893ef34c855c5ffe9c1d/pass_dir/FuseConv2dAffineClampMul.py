import torch
import triton
import triton.language as tl


# Autotune configurations
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 1, 'BLOCK_W': 1}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_H': 2, 'BLOCK_W': 2}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 4}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8}, num_stages=1, num_warps=2),
    ],
    key=['H', 'W'],
)
@triton.jit
def fused_conv2d_affine_clamp_mul_kernel(
    in_0_ptr,  # bias [C]
    in_1_ptr,  # weight [OC, IC, 1, 1]
    in_2_ptr,  # input features [B, C, H, W]
    in_3_ptr,  # conv input [B, IC, 1, 1]
    out_ptr,
    # Output shape
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    # Weight shape
    OC: tl.constexpr, IC: tl.constexpr,
    # Block sizes for autotune
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Use 2D grid: (B, C) - one program per output channel per batch
    b = tl.program_id(0)
    c = tl.program_id(1)
    
    # Compute 1x1 conv: result[b, c] = sum_ic(in[b, ic] * weight[c, ic]) + bias[c]
    conv_result = 0.0
    
    # Load bias
    bias_val = tl.load(in_0_ptr + c).to(tl.float32)
    
    # Compute conv sum over input channels (vectorized load if possible)
    for ic in range(IC):
        # Load input[b, ic]
        in_offset = b * IC + ic
        in_val = tl.load(in_3_ptr + in_offset).to(tl.float32)
        
        # Load weight[c, ic]
        weight_offset = c * IC + ic
        weight_val = tl.load(in_1_ptr + weight_offset).to(tl.float32)
        
        conv_result = conv_result + in_val * weight_val
    
    conv_result = conv_result + bias_val
    
    # Apply affine transform: (x + 1.0) / 2.0
    transformed = (conv_result + 1.0) / 2.0
    
    # Clamp to [0, 1]
    clamped = transformed if (transformed >= 0.0 and transformed <= 1.0) else (0.0 if transformed < 0.0 else 1.0)
    
    # Now expand to all spatial positions (H * W) and multiply with in_2
    # Use blocked tiling for better memory coalescing
    h_start = 0
    w_start = 0
    
    # Process in blocks for better memory access pattern
    for h_idx in range(H):
        for w_idx in range(W):
            # Load in_2[b, c, h, w]
            in2_offset = ((b * C + c) * H + h_idx) * W + w_idx
            in2_val = tl.load(in_2_ptr + in2_offset).to(tl.float32)
            
            # Multiply and store
            result = clamped * in2_val
            out_offset = ((b * C + c) * H + h_idx) * W + w_idx
            tl.store(out_ptr + out_offset, result)


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: conv2d + add + div + clamp + mul
    """
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d + 1.0
    tmp_4 = tmp_3 / 2.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract shape and type information for the optimized kernel
    """
    # Get shapes
    bias_shape = in_0.shape      # [C]
    weight_shape = in_1.shape    # [OC, IC, 1, 1]
    in2_shape = in_2.shape       # [B, C, H, W]
    conv_in_shape = in_3.shape   # [B, IC, 1, 1]
    
    return (in_0, in_1, in_2, in_3, bias_shape, weight_shape, in2_shape, conv_in_shape)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2, in_3, bias_shape, weight_shape, in2_shape, conv_in_shape):
    """
    Wrapper function that launches the Triton kernel
    """
    B, C, H, W = in2_shape
    OC, IC = weight_shape[0], weight_shape[1]
    
    # Allocate output tensor
    out = torch.empty((B, C, H, W), dtype=in_2.dtype, device=in_2.device)
    
    # 2D grid: (B, C) - each program handles one (batch, channel) pair
    grid = (B, C)
    
    fused_conv2d_affine_clamp_mul_kernel[grid](
        in_0, in_1, in_2, in_3, out,
        B, C, H, W,
        OC, IC,
    )
    
    return out


def replacement_func():
    return fused_kernel_wrapper