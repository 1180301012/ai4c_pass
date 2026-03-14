import torch
import triton
import triton.language as tl


def pattern(in_0, in_3, in_4, in_5, in_6, in_7, in_8):
    """
    Match: mul(gamma) -> add(residual) -> batch_norm
    This fuses 3 element-wise operations into a single kernel.
    The conv2d and dropout are kept in the graph, only the element-wise ops are fused.
    """
    tmp_0 = in_0  # gamma [64, 1, 1]
    tmp_3 = in_3  # bn running_mean [64]
    tmp_4 = in_4  # bn running_var [64]
    tmp_5 = in_5  # bn bias [64]
    tmp_6 = in_6  # bn weight [64]
    tmp_7 = in_7  # residual [B, 64, H, W]
    tmp_8 = in_8  # after_dropout [B, 64, H, W] - output of dropout or conv
    
    # Multiply by gamma
    tmp_scaled = tmp_8 * tmp_0
    
    # Add residual
    tmp_add = tmp_7 + tmp_scaled
    
    # BatchNorm - returns output tensor only when training=False
    tmp_bn = torch.nn.functional.batch_norm(tmp_add, tmp_3, tmp_4, tmp_6, tmp_5, False, 0.1, 1e-05)
    
    return tmp_bn, tmp_add


def replacement_args(in_0, in_3, in_4, in_5, in_6, in_7, in_8):
    return (in_0, in_3, in_4, in_5, in_6, in_7, in_8)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=8),
    ],
    key=['BLOCK_SIZE'],
)
@triton.jit
def fused_mul_add_bn_kernel(
    # Input pointers
    conv_out_ptr, gamma_ptr, residual_ptr,
    # BN parameters
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    # Output
    output_ptr, pre_bn_ptr,
    # Dimensions
    B, C, H, W,
    N,
    # Meta
    BLOCK_SIZE: tl.constexpr
):
    """Fused kernel for: mul(gamma) -> add(residual) -> batch_norm"""
    
    # Each program processes BLOCK_SIZE elements
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    
    # Calculate b, c, h, w from flattened offset
    elements_per_channel = H * W
    elements_per_batch = C * H * W
    
    b = offset // elements_per_batch
    rem = offset % elements_per_batch
    c = rem // elements_per_channel
    rem = rem % elements_per_channel
    h = rem // W
    w = rem % W
    
    # Load conv_out
    conv_out = tl.load(conv_out_ptr + offset, mask=mask, other=0.0)
    
    # Load gamma and multiply
    gamma_val = tl.load(gamma_ptr + c, mask=mask, other=1.0)
    scaled = conv_out * gamma_val
    
    # Load residual and add
    residual_offset = b * (C * H * W) + c * (H * W) + h * W + w
    res = tl.load(residual_ptr + residual_offset, mask=mask, other=0.0)
    pre_bn = scaled + res
    
    # Store pre-BN result
    tl.store(pre_bn_ptr + offset, pre_bn, mask=mask)
    
    # Apply batch norm: (x - mean) / sqrt(var + eps) * weight + bias
    mean = tl.load(bn_mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(bn_var_ptr + c, mask=mask, other=1.0)
    bn_w = tl.load(bn_weight_ptr + c, mask=mask, other=1.0)
    bn_b = tl.load(bn_bias_ptr + c, mask=mask, other=0.0)
    
    eps = 1e-05
    inv_std = 1.0 / tl.sqrt(var + eps)
    normalized = (pre_bn - mean) * inv_std * bn_w + bn_b
    
    # Store output
    tl.store(output_ptr + offset, normalized, mask=mask)


def fused_mul_add_bn(gamma, conv_out, residual, bn_mean, bn_var, bn_weight, bn_bias):
    """Fused kernel: multiply(gamma) + add(residual) + batch_norm"""
    B, C, H, W = conv_out.shape
    N = B * C * H * W
    
    # Allocate output tensors
    bn_output = torch.empty_like(conv_out)
    pre_bn = torch.empty_like(conv_out)
    
    # Calculate grid
    grid = (triton.cdiv(N, 1024),)
    
    fused_mul_add_bn_kernel[grid](
        conv_out.reshape(-1),
        gamma.squeeze(),  # [64, 1, 1] -> [64]
        residual.reshape(-1),
        bn_mean, bn_var, bn_weight, bn_bias,
        bn_output.reshape(-1), pre_bn.reshape(-1),
        B, C, H, W, N,
        BLOCK_SIZE=1024,
    )
    
    return bn_output, pre_bn


@torch.fx.wrap
def kernel_wrapper(gamma, bn_mean, bn_var, bn_weight, bn_bias, residual, conv_out):
    return fused_mul_add_bn(gamma, conv_out, residual, bn_mean, bn_var, bn_weight, bn_bias)


def replacement_func():
    return kernel_wrapper