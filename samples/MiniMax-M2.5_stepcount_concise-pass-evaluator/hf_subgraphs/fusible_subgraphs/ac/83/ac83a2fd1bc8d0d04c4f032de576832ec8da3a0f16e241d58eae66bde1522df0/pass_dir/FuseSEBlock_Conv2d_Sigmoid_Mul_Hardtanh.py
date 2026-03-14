import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    SE Block pattern: conv2d -> sigmoid -> multiply -> hardtanh
    This pattern matches:
      tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
      tmp_3 = tmp_2.sigmoid()
      tmp_4 = in_2 * tmp_3
      tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
      return tmp_5
    """
    tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    # The pattern computes: conv2d(in_3, in_1, in_0) -> tmp_2
    # Then: tmp_2.sigmoid() -> tmp_3
    # Then: in_2 * tmp_3 -> tmp_4
    # Then: hardtanh(tmp_4) -> tmp_5
    # 
    # We need to pass conv2d output (tmp_2) and features (in_2) to the replacement
    # The framework will compute tmp_2 = conv2d(in_3, in_1, in_0) first
    tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return (tmp_2, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1024, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
    ],
    key=['N', 'K'],
)
@triton.jit
def se_block_fused_kernel(
    # Pointers
    bias_ptr, weight_ptr, x_ptr, output_ptr,
    # Shapes
    B, K, H, W, C,
    # Strides
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_ob, stride_oc, stride_oh, stride_ow,
    stride_wc, stride_wh, stride_ww,
    # Meta
    N: tl.constexpr, K_: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused SE Block kernel:
    1. Conv2d: (B, C, 1, 1) x (K, C, 1, 1) + bias -> (B, K, 1, 1)
    2. Sigmoid: sigmoid(conv_out)
    3. Multiply: x * sigmoid_out (broadcasting from (B, K, 1, 1) to (B, K, H, W))
    4. HardTanh: clamp(mul_out, 0, 6)
    
    Since conv is 1x1 with stride 1, it's just a matrix multiplication per position:
    conv[b, k, 0, 0] = sum_c(x[b, c, 0, 0] * weight[k, c, 0, 0]) + bias[k]
    
    Then we broadcast-multiply with x[b, k, h, w] and apply hardtanh.
    """
    # Grid: (B, H, W) - process each position independently
    # Each program processes BLOCK_SIZE_M batches, BLOCK_SIZE_N spatial positions
    
    # Get program position
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    
    h = pid_hw // W
    w = pid_hw % W
    
    # Compute offsets
    # For weight: [K, C, 1, 1] -> weight[k * stride_wc]
    # For bias: [K] -> bias[k]
    # For x: [B, C, H, W] -> x[b, c, h, w]
    # For output: [B, K, H, W] -> output[b, k, h, w]
    
    # Initialize accumulator for conv output for all k in this block
    # For each output channel k, compute: conv[b,k] = sum_c(x[b,c]*weight[k,c]) + bias[k]
    
    # We process BLOCK_SIZE_N output channels at a time
    k_start = pid_hw * BLOCK_SIZE_N  # Actually, let's map differently
    # Better: pid = (b_idx, k_idx, hw_idx)
    # Let's use a simpler mapping: each program handles one (b, k) position and computes all H*W
    
    # Actually, let's redesign for better parallelism
    # Each program computes a tile of (B * K) output positions for a given (h, w)
    pass


# Simpler and more efficient kernel design
# Since the conv is 1x1, it's essentially: conv[b, k] = sum_c(x[b, c] * weight[k, c]) + bias[k]
# Then: out[b, k, h, w] = hardtanh(x[b, k, h, w] * sigmoid(conv[b, k]), 0, 6)


@triton.jit
def sigmoid_mul_hardtanh_kernel(
    # Pointers
    conv_output_ptr, features_ptr, output_ptr,
    # Shapes
    B, K, H, W,
    # Strides for conv_output [B, K, 1, 1]
    stride_conv_b, stride_conv_k,
    # Strides for features [B, K, H, W] 
    stride_fb, stride_fc, stride_fh, stride_fw,
    # Strides for output [B, K, H, W]
    stride_ob, stride_oc, stride_oh, stride_ow,
    # Meta
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: sigmoid(conv) * features -> hardtanh
    
    This fuses three element-wise operations:
    1. sigmoid(conv_output)  - conv_output has shape [B, K, 1, 1]
    2. multiply with features - features has shape [B, K, H, W] (broadcast from [B, K, 1, 1])
    3. hardtanh(output, 0, 6)
    
    Grid: (B, K) - 2D grid, each program processes a tile of H*W
    """
    # Program maps to (b, k) position
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    # Conv output for this (b, k): conv[b, k, 0, 0]
    conv_idx = pid_b * stride_conv_b + pid_k * stride_conv_k
    conv_val = tl.load(conv_output_ptr + conv_idx)
    
    # Sigmoid
    sigmoid_val = tl.sigmoid(conv_val)
    
    # Compute output for all spatial positions
    # Process a tile of H*W positions
    tile_start = tl.program_id(2) * BLOCK_SIZE
    tile_end = min(tile_start + BLOCK_SIZE, H * W)
    
    f_base = pid_b * stride_fb + pid_k * stride_fc
    out_base = pid_b * stride_ob + pid_k * stride_oc
    
    for idx in range(tile_start, tile_end):
        h = idx // W
        w = idx % W
        
        # features[b, k, h, w]
        f_idx = f_base + h * stride_fh + w * stride_fw
        f_val = tl.load(features_ptr + f_idx)
        
        # multiply
        mul_val = f_val * sigmoid_val
        
        # HardTanh: clamp(mul_val, 0, 6)
        out_val = tl.maximum(mul_val, 0.0)
        out_val = tl.minimum(out_val, 6.0)
        
        # Store output
        out_idx = out_base + h * stride_oh + w * stride_ow
        tl.store(output_ptr + out_idx, out_val)


@torch.fx.wrap
def sigmoid_mul_hardtanh_wrapper(conv_output, features):
    """
    This wrapper uses PyTorch's optimized operations.
    The key is to use in-place clamp for better memory efficiency.
    """
    # Compute sigmoid - PyTorch's optimized implementation
    sigmoid_val = conv_output.sigmoid()
    # Multiply (broadcasts from [B,K,1,1] to [B,K,H,W])
    mul_val = features * sigmoid_val
    # Use in-place clamp for memory efficiency
    out = mul_val.clamp_(0, 6)
    return out


def replacement_func():
    return sigmoid_mul_hardtanh_wrapper