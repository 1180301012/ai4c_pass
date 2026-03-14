import torch
import triton
import triton.language as tl


# Pattern matching function - matches Add + BatchNorm + ReLU
def pattern(in_1, in_2, in_3, in_4, in_7, in_8):
    """
    Match the pattern:
    tmp_8 = in_7 + tmp_7  (from interpolate)
    tmp_9 = torch.nn.functional.batch_norm(tmp_8, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    tmp_10 = torch.nn.functional.relu(tmp_9, inplace=True)
    
    Returns tmp_10 (final output)
    """
    tmp_8 = in_7 + in_8
    tmp_9 = torch.nn.functional.batch_norm(tmp_8, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    tmp_10 = torch.nn.functional.relu(tmp_9, inplace=True)
    return tmp_10


def replacement_args(in_1, in_2, in_3, in_4, in_7, in_8):
    return (in_1, in_2, in_3, in_4, in_7, in_8)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def bn_relu_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N, C, H, W,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused BatchNorm + ReLU kernel.
    
    BatchNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
    ReLU: y = max(0, x)
    
    Both can be fused efficiently.
    """
    # Get program IDs
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Offsets
    off_n = pid_n
    off_c = pid_c
    
    # Load BN parameters for this channel
    # running_mean: [C]
    mean = tl.load(running_mean_ptr + off_c)
    # running_var: [C]
    var = tl.load(running_var_ptr + off_c)
    # weight: [C]
    weight = tl.load(weight_ptr + off_c)
    # bias: [C]
    bias = tl.load(bias_ptr + off_c)
    
    # Compute inverse std
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Normalization factors
    weight_scaled = weight * inv_std
    bias_adjusted = bias - mean * weight * inv_std
    
    # Process spatial positions
    for h_start in range(0, H, BLOCK_M):
        for w_start in range(0, W, BLOCK_N):
            # Offsets
            h_offs = h_start + tl.arange(0, BLOCK_M)
            w_offs = w_start + tl.arange(0, BLOCK_N)
            
            # Mask
            mask_h = h_offs < H
            mask_w = w_offs < W
            mask = mask_h & mask_w
            
            # Load input [N, C, H, W]
            input_offsets = (off_n * C * H * W + 
                             off_c * H * W + 
                             h_offs[:, None] * W + 
                             w_offs[None, :])
            x = tl.load(input_ptr + input_offsets, mask=mask_h[:, None] & mask_w[None, :], other=0.0)
            
            # BatchNorm: (x - mean) * weight_scaled + bias_adjusted
            y = x * weight_scaled + bias_adjusted
            
            # ReLU: max(0, x)
            y = tl.maximum(y, 0.0)
            
            # Store output
            tl.store(output_ptr + input_offsets, y, mask=mask_h[:, None] & mask_w[None, :])


def fused_bn_relu(running_mean, running_var, weight, bias, input):
    """
    Fused BatchNorm + ReLU operation.
    
    Args:
        running_mean: [C] - Running mean
        running_var: [C] - Running variance
        weight: [C] - BN weight (gamma)
        bias: [C] - BN bias (beta)
        input: [N, C, H, W] - Input tensor
    
    Returns:
        [N, C, H, W] - Output tensor
    """
    N, C, H, W = input.shape
    eps = 1e-05
    
    # Output
    output = torch.empty_like(input)
    
    # Grid: (N, C)
    grid = (N, C)
    
    bn_relu_kernel[grid](
        input,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        N, C, H, W,
        eps,
    )
    
    return output


@torch.fx.wrap
def fused_bn_relu_wrapper(running_mean, running_var, weight, bias, input):
    return fused_bn_relu(running_mean, running_var, weight, bias, input)


def replacement_func():
    return fused_bn_relu_wrapper