import torch
import triton
import triton.language as tl


def pattern(in_6, tmp_4, in_0, in_1, in_3, in_2, in_5):
    """
    Pattern: Conv2D + BatchNorm + Add (residual connection)
    This matches the exact computation from model.py:
    - tmp_5 = torch.conv2d(in_6, tmp_4, None, (1, 1), (0, 0), (1, 1), 1)
    - tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    - tmp_6 += in_5
    """
    tmp_5 = torch.conv2d(in_6, tmp_4, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = tmp_6 + in_5
    return tmp_5, tmp_6


def replacement_args(in_6, tmp_4, in_0, in_1, in_3, in_2, in_5):
    """
    Extract arguments needed for the fused kernel:
    - in_6: input feature map (N, C_in, H, W)
    - tmp_4: conv weight (C_out, C_in, 1, 1)
    - in_0: running mean
    - in_1: running var
    - in_3: batch norm weight (gamma)
    - in_2: batch norm bias (beta)
    - in_5: residual connection
    """
    return (in_6, tmp_4, in_0, in_1, in_3, in_2, in_5)


@triton.autotune(
    configs=[
        # Different tile sizes for various input sizes
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def fused_conv_bn_add_kernel(
    # Input pointers
    input_ptr, weight_ptr, mean_ptr, var_ptr, weight_bn_ptr, bias_bn_ptr, residual_ptr,
    # Output pointer
    output_ptr,
    # Dimensions
    N, C_in, C_out, H, W,
    # Strides
    stride_input_n, stride_input_c, stride_input_h, stride_input_w,
    stride_weight_c, stride_weight_cin,
    stride_residual_n, stride_residual_c, stride_residual_h, stride_residual_w,
    stride_output_n, stride_output_c, stride_output_h, stride_output_w,
    # BN parameters
    eps: tl.constexpr,
    # Block config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused kernel: Conv2D (1x1) + BatchNorm + Add
    Computes: output = (((input @ weight) - mean) / sqrt(var + eps)) * weight_bn + bias_bn + residual
    """
    # Get program IDs
    pid = tl.program_id(0)
    num_pid_n = N
    
    # Calculate which N this program handles
    pid_n = pid % num_pid_n
    pid_group = pid // num_pid_n
    
    # Calculate the offset in the output tensor
    # Each program processes a portion of the batch
    off_n = pid_n
    
    # Load mean, var, weight_bn, bias_bn (they're 1D tensors of size C_out)
    # Use program_id to distribute loading across programs
    mean_offset = pid_group * BLOCK_N
    var_offset = pid_group * BLOCK_N
    weight_bn_offset = pid_group * BLOCK_N
    bias_bn_offset = pid_group * BLOCK_N
    
    # Calculate the number of channels each program handles
    num_channels = C_out
    num_channel_groups = (num_channels + BLOCK_N - 1) // BLOCK_N
    num_n_groups = (N + BLOCK_M - 1) // BLOCK_M
    
    # Each program processes one channel group
    channel_group_id = pid_group % num_channel_groups
    batch_group_id = pid_group // num_channel_groups
    
    # Offset for channels
    channel_start = channel_group_id * BLOCK_N
    channel_end = min(channel_start + BLOCK_N, C_out)
    channel_mask = channel_end - channel_start
    
    # Offset for batch
    batch_start = batch_group_id * BLOCK_M
    batch_end = min(batch_start + BLOCK_M, N)
    batch_mask = batch_end - batch_start
    
    # Load BN parameters for this channel range
    mean = tl.load(mean_ptr + channel_start + tl.arange(0, BLOCK_N)[:channel_mask])
    var = tl.load(var_ptr + channel_start + tl.arange(0, BLOCK_N)[:channel_mask])
    weight_bn = tl.load(weight_bn_ptr + channel_start + tl.arange(0, BLOCK_N)[:channel_mask])
    bias_bn = tl.load(bias_bn_ptr + channel_start + tl.arange(0, BLOCK_N)[:channel_mask])
    
    # Compute normalization factor: 1 / sqrt(var + eps)
    norm_factor = tl.rsqrt(var + eps)
    
    # Normalize: (x - mean) * norm_factor
    normalized = (mean * (-norm_factor))
    normalized = normalized + (norm_factor * mean)  # This cancels out, let me simplify
    # Actually: (x - mean) / sqrt(var + eps) = x * rsqrt(var+eps) - mean * rsqrt(var+eps)
    
    # Initialize output accumulator for this program
    output_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate over spatial dimensions (H x W)
    for h in range(H):
        for w in range(W):
            # Load input feature: shape (BLOCK_M, C_in)
            # input[off_n, :, h, w]
            input_indices = (
                off_n * stride_input_n + 
                tl.arange(0, BLOCK_M)[:, None] * stride_input_n +
                tl.arange(0, C_in)[None, :] * stride_input_c +
                h * stride_input_h +
                w * stride_input_w
            )
            
            # This is complex for 1x1 conv, let's simplify:
            # For 1x1 conv: output[n, c_out, h, w] = sum_c input[n, c, h, w] * weight[c_out, c]
            # This is essentially a matrix multiply for each spatial location
            
            # Load input slice for this spatial location: (C_in,)
            for c_in in range(C_in):
                input_val = tl.load(
                    input_ptr + 
                    off_n * stride_input_n +
                    c_in * stride_input_c +
                    h * stride_input_h +
                    w * stride_input_w
                )
                
                # Load weight slice: (C_out,)
                weight_slice = tl.load(
                    weight_ptr + 
                    channel_start * stride_weight_c +
                    c_in * stride_weight_cin +
                    tl.arange(0, BLOCK_N)[:channel_mask]
                )
                
                # Accumulate conv result
                output_acc += input_val * weight_slice
            
            # Apply BN and add residual
            # output = conv_result * weight_bn * rsqrt(var + eps) + bias_bn - mean * weight_bn * rsqrt(var + eps)
            # Simplified: output = (conv_result - mean) * weight_bn * rsqrt(var + eps) + bias_bn
            
            # Compute final output for this spatial location
            bn_scaled = output_acc * weight_bn * norm_factor
            final_out = bn_scaled + bias_bn
            
            # Load residual
            residual = tl.load(
                residual_ptr + 
                off_n * stride_residual_n +
                channel_start * stride_residual_c +
                h * stride_residual_h +
                w * stride_residual_w +
                tl.arange(0, BLOCK_N)[:channel_mask]
            )
            
            # Add residual
            final_out = final_out + residual
            
            # Store result
            tl.store(
                output_ptr + 
                off_n * stride_output_n +
                channel_start * stride_output_c +
                h * stride_output_h +
                w * stride_output_w +
                tl.arange(0, BLOCK_N)[:channel_mask],
                final_out
            )
            
            # Reset accumulator
            output_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


def fused_conv_bn_add(x, weight, mean, var, weight_bn, bias_bn, residual):
    """
    Wrapper function for the fused Conv2D + BatchNorm + Add kernel.
    """
    # Get dimensions
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    
    # Output shape matches residual shape
    output = torch.empty_like(residual)
    
    # Calculate grid
    # We parallelize over (N * C_out) with channel blocking
    num_channel_groups = (C_out + 255) // 256
    num_batch_groups = (N + 15) // 16
    num_programs = num_channel_groups * num_batch_groups
    
    # For smaller tensors, use simpler config
    if N * C_out <= 1024:
        grid = (N * C_out,)
    else:
        grid = (num_programs,)
    
    # Launch kernel
    fused_conv_bn_add_kernel[grid](
        x, weight, mean, var, weight_bn, bias_bn, residual, output,
        N, C_in, C_out, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1),
        residual.stride(0), residual.stride(1), residual.stride(2), residual.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        1e-05,
    )
    
    return output


# Need to wrap with torch.fx.wrap for FX tracing
fused_conv_bn_add = torch.fx.wrap(fused_conv_bn_add)


def replacement_func():
    """
    Returns the fused kernel function.
    """
    return fused_conv_bn_add