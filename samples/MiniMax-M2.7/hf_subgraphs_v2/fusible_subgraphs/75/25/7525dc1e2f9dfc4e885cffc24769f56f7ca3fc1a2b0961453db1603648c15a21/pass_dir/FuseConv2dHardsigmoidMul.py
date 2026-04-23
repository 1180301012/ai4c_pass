import torch
import triton
import triton.language as tl

# Pattern matching function - matches conv2d + hardsigmoid + mul
def pattern(in_0, in_1, in_2, in_3):
    """
    Match the SE block pattern: conv2d + hardsigmoid + mul
    The pattern matches:
      conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
      tmp_3 = torch.nn.functional.hardsigmoid(conv2d, False)
      tmp_4 = in_2 * tmp_3
    """
    conv2d_result = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    hs_result = torch.nn.functional.hardsigmoid(conv2d_result, False)
    mul_result = in_2 * hs_result
    return mul_result


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments needed for the fused kernel.
    in_0: bias tensor [1024]
    in_1: weight tensor [1024, 1024, 1, 1]
    in_2: input tensor [batch, 1024, H, W] 
    in_3: input tensor [batch, 1024, 1, 1]
    """
    return (in_0, in_1, in_2, in_3)


# Optimized Triton kernel for fused conv2d + hardsigmoid + mul
# Strategy: Use 2D grid where grid_x handles channels (parallel) and
# grid_y handles batch elements. Each program computes the conv result
# for its channel by reducing over input channels, then broadcasts to spatial.

@triton.jit
def fused_conv_hardsigmoid_mul_kernel(
    # Tensor pointers
    bias_ptr, weight_ptr, in2_ptr, in3_ptr, out_ptr,
    # Tensor strides (for 1D access)
    bias_stride_b, 
    weight_stride_cout, weight_stride_cin, weight_stride_h, weight_stride_w,
    in2_stride_b, in2_stride_c, in2_stride_h, in2_stride_w,
    in3_stride_b, in3_stride_c, in3_stride_h, in3_stride_w,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    # Shape info
    batch_size, num_channels, out_h, out_w,
    # Block size for channel reduction
    CHANNEL_BLOCK_SIZE: tl.constexpr
):
    # 2D grid: (num_channels, batch_size)
    pid_c = tl.program_id(0)  # Channel dimension
    pid_b = tl.program_id(1)  # Batch dimension
    
    # Compute 1x1 convolution: conv[b, c] = bias[c] + sum_k(weight[c,k] * in3[b,k])
    # For 1x1 conv with 1x1 input, we just do the matmul
    conv_val = tl.load(bias_ptr + pid_c * bias_stride_b)
    
    # Reduce over input channels in blocks
    num_cin = num_channels
    for ki in range(0, num_cin, CHANNEL_BLOCK_SIZE):
        k_end = min(ki + CHANNEL_BLOCK_SIZE, num_cin)
        
        # Create offsets for this chunk
        k_offsets = tl.arange(0, CHANNEL_BLOCK_SIZE)
        mask = k_offsets < (k_end - ki)
        
        # Load weights: weight[c_out, k] - c_out is fixed at pid_c
        weight_offset = pid_c * weight_stride_cout + (ki + k_offsets) * weight_stride_cin
        w_vals = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
        
        # Load in3: in3[b, k] - b is fixed at pid_b
        in3_offset = pid_b * in3_stride_b + (ki + k_offsets) * in3_stride_c
        x_vals = tl.load(in3_ptr + in3_offset, mask=mask, other=0.0)
        
        # Accumulate
        conv_val = conv_val + tl.sum(w_vals * x_vals)
    
    # Apply hardsigmoid: hs(x) = relu6(x + 3) / 6
    hs_val = conv_val + 3.0
    hs_val = tl.where(hs_val > 0.0, hs_val, 0.0)
    hs_val = tl.where(hs_val < 6.0, hs_val, 6.0)
    hs_val = hs_val * (1.0 / 6.0)
    
    # Compute output offset base: [b, c, 0, 0]
    out_base = pid_b * out_stride_b + pid_c * out_stride_c
    in2_base = pid_b * in2_stride_b + pid_c * in2_stride_c
    
    # Iterate over spatial positions
    for h_idx in range(out_h):
        for w_idx in range(out_w):
            # Load in2 value
            in2_offset = in2_base + h_idx * in2_stride_h + w_idx * in2_stride_w
            in2_val = tl.load(in2_ptr + in2_offset)
            
            # Store result
            out_offset = out_base + h_idx * out_stride_h + w_idx * out_stride_w
            tl.store(out_ptr + out_offset, in2_val * hs_val)


@torch.fx.wrap
def fused_conv_hardsigmoid_mul_wrapper(bias, weight, in2, in3):
    """
    Wrapper function for the fused conv2d + hardsigmoid + mul kernel.
    """
    batch_size, channels, out_h, out_w = in2.shape
    
    # Allocate output tensor
    out = torch.empty_like(in2)
    
    # Grid: (channels, batch_size) - each (c, b) pair computes one channel output for one batch
    grid = (channels, batch_size)
    
    # Channel block size for reduction
    CHANNEL_BLOCK_SIZE = 32
    
    fused_conv_hardsigmoid_mul_kernel[grid](
        bias, weight, in2, in3, out,
        bias.stride(0),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        in2.stride(0), in2.stride(1), in2.stride(2), in2.stride(3),
        in3.stride(0), in3.stride(1), in3.stride(2), in3.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch_size, channels, out_h, out_w,
        CHANNEL_BLOCK_SIZE=CHANNEL_BLOCK_SIZE
    )
    
    return out


def replacement_func():
    """
    Returns the fused kernel function.
    """
    return fused_conv_hardsigmoid_mul_wrapper