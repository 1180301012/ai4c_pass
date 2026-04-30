import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv_bn_add_kernel_v2(
    # Conv inputs
    input_ptr, weight_ptr,
    # BN inputs (running_mean, running_var, weight, bias)
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    # Residual input (add tensor)
    residual_ptr,
    # Output
    output_ptr,
    # Shape info
    N, C_in, C_out, H, W,
    # BN epsilon
    bn_eps: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Conv2d (1x1) + BatchNorm + Add kernel for resnet10t pattern.
    This kernel performs:
        output = batch_norm(conv(input, weight)) + residual
    """
    # Get program ID and compute flat index
    pid = tl.program_id(0)
    total_elements = N * C_out * H * W
    
    # Bounds check
    if pid * BLOCK_SIZE >= total_elements:
        return
    
    # Compute flat indices for this block
    flat_indices = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = flat_indices < total_elements
    
    # Compute N, C, H, W indices from flat index
    # Layout: (N, C_out, H, W)
    n = flat_indices // (C_out * H * W)
    remaining = flat_indices % (C_out * H * W)
    c_out = remaining // (H * W)
    h = (remaining % (H * W)) // W
    w = remaining % W
    
    # Conv2d (1x1, stride=1, padding=0) is just a channel-wise operation
    # output[n, c_out, h, w] = sum_c_in(input[n, c_in, h, w] * weight[c_out, c_in, 0, 0])
    # For resnet10t: weight shape [128, 64, 1, 1]
    
    # Load running mean and var for this channel
    running_mean = tl.load(bn_mean_ptr + c_out)
    running_var = tl.load(bn_var_ptr + c_out)
    gamma = tl.load(bn_weight_ptr + c_out)
    beta = tl.load(bn_bias_ptr + c_out)
    
    # Compute inv_std for batch norm
    inv_std = tl.rsqrt(running_var + bn_eps)
    
    # Conv2d 1x1: need to sum over input channels
    conv_result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for c_in in range(C_in):
        # Compute weight flat index: [c_out, c_in, 0, 0]
        weight_idx = c_out * C_in + c_in
        w_val = tl.load(weight_ptr + weight_idx)
        
        # Input flat index: [n, c_in, h, w]
        input_idx = n * C_in * H * W + c_in * H * W + h * W + w
        input_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        
        conv_result += input_val * w_val
    
    # Normalize with batch norm
    # (x - mean) * inv_std * gamma + beta
    norm_result = (conv_result - running_mean) * inv_std * gamma + beta
    
    # Load residual and add
    residual = tl.load(residual_ptr + flat_indices, mask=mask, other=0.0)
    output = norm_result + residual
    
    # Store result
    tl.store(output_ptr + flat_indices, output, mask=mask)


@torch.fx.wrap
def fused_conv_bn_add_v2_wrapper(
    input, weight,
    bn_mean, bn_var, bn_weight, bn_bias,
    residual,
    output,
    bn_eps=1e-05
):
    """
    Wrapper for the fused Conv2d + BatchNorm + Add kernel (resnet10t variant).
    """
    N, C_in, H, W = input.shape
    C_out, _, _, _ = weight.shape
    _, _, H_out, W_out = residual.shape
    
    # Total elements in output
    total_elements = N * C_out * H_out * W_out
    
    # Choose block size
    BLOCK_SIZE = 256
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_conv_bn_add_kernel_v2[(num_programs,)](
        input, weight,
        bn_mean, bn_var, bn_weight, bn_bias,
        residual,
        output,
        N, C_in, C_out, H_out, W_out,
        bn_eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# Pattern for graphs where conv uses in_5 and in_0, BN uses in_1, in_2, in_4, in_3, add uses in_6
# Pattern: conv2d(in_5, in_0) -> batch_norm -> (in_6 += result)
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Match pattern: conv2d(in_5, in_0) + batch_norm + in_6
    This is the pattern found in resnet10t graphs.
    Note: in_6 += tmp_6 (in-place add)
    """
    conv_result = torch.conv2d(in_5, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    bn_result = torch.nn.functional.batch_norm(conv_result, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    # Note: in_6 += bn_result (in-place)
    in_6 += bn_result
    return in_6


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


def replacement_func():
    return fused_conv_bn_add_v2_wrapper