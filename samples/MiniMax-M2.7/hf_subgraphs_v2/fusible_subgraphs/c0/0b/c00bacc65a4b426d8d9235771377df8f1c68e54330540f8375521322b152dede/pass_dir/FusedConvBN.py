import torch
import triton
import triton.language as tl

"""
Fused Conv2d + BatchNorm kernel for GPU optimization.

This pass fuses the convolution and batch normalization operations during inference.
The batch_norm with training=False uses running statistics (constants), allowing
us to fuse the affine transformation directly into the convolution.
"""

def pattern(in_5, in_4, in_0, in_1, in_3, in_2):
    """
    Match conv2d + batch_norm pattern.
    
    Args:
        in_5: Input tensor for convolution [N, C_in, H, W]
        in_4: Convolution weight [C_out, C_in, K, K]
        in_0: Running mean [C_out]
        in_1: Running variance [C_out]
        in_3: BN weight (gamma) [C_out]
        in_2: BN bias (beta) [C_out]
    
    Returns:
        Normalized tensor after batch_norm [N, C_out, H_out, W_out]
    """
    conv2d = torch.conv2d(in_5, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_6

def replacement_args(in_5, in_4, in_0, in_1, in_3, in_2):
    return (in_5, in_4, in_0, in_1, in_3, in_2)

@triton.jit
def fused_conv_bn_kernel(
    x_ptr, weight_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr, output_ptr,
    N, C_in, C_out, H, W, K,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_w_cout, stride_w_cin, stride_w_kh, stride_w_kw,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    eps: tl.constexpr, USE_FP32: tl.constexpr
):
    """
    Fused Conv2d + BatchNorm kernel with improved numerical stability.
    
    For fp16/bf16 inputs, we compute in fp32 and cast back to maintain precision.
    Each program computes one output element (n, c_out, h, w).
    """
    pid = tl.program_id(0)
    
    # Decode output position
    n = pid // (C_out * H * W)
    tmp = pid % (C_out * H * W)
    c_out = tmp // (H * W)
    tmp = tmp % (H * W)
    h = tmp // W
    w = tmp % W
    
    # Load and compute BN parameters in fp32 for numerical stability
    mean_f = tl.load(mean_ptr + c_out).to(tl.float32)
    var_f = tl.load(var_ptr + c_out).to(tl.float32)
    gamma_f = tl.load(gamma_ptr + c_out).to(tl.float32)
    beta_f = tl.load(beta_ptr + c_out).to(tl.float32)
    
    std_f = tl.sqrt(var_f + eps)
    scale_f = gamma_f / std_f
    bias_f = beta_f - gamma_f * mean_f / std_f
    
    # Convolution accumulation in fp32
    conv_out = 0.0
    pad = K // 2
    
    for c_in in range(C_in):
        for kh in range(K):
            for kw in range(K):
                h_in = h + kh - pad
                w_in = w + kw - pad
                
                # Boundary check (for non-square or edge cases)
                if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                    x_idx = n * stride_x_n + c_in * stride_x_c + h_in * stride_x_h + w_in * stride_x_w
                    w_idx = c_out * stride_w_cout + c_in * stride_w_cin + kh * stride_w_kh + kw * stride_w_kw
                    
                    x_val = tl.load(x_ptr + x_idx).to(tl.float32)
                    w_val = tl.load(weight_ptr + w_idx).to(tl.float32)
                    conv_out += x_val * w_val
    
    # Apply BN: y = scale * conv_out + bias
    out_val = conv_out * scale_f + bias_f
    
    # Cast back to the original dtype
    if USE_FP32:
        tl.store(output_ptr + n * stride_out_n + c_out * stride_out_c + h * stride_out_h + w * stride_out_w, out_val)
    else:
        tl.store(output_ptr + n * stride_out_n + c_out * stride_out_c + h * stride_out_h + w * stride_out_w, out_val.to(tl.float16))

@torch.fx.wrap
def fused_conv_bn_impl(x, weight, mean, var, gamma, beta):
    """Wrapper for the fused conv+bn kernel."""
    N, C_in, H, W = x.shape
    C_out, _, K, _ = weight.shape
    H_out, W_out = H, W  # For stride=1, pad=(K-1)//2
    
    output = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    total_elements = N * C_out * H_out * W_out
    
    BLOCK_SIZE = 256
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Determine if we need fp32 computation
    use_fp32 = x.dtype in (torch.float16, torch.bfloat16)
    
    fused_conv_bn_kernel[grid](
        x, weight, mean, var, gamma, beta, output,
        N, C_in, C_out, H, W, K,
        x.stride()[0], x.stride()[1], x.stride()[2], x.stride()[3],
        weight.stride()[0], weight.stride()[1], weight.stride()[2], weight.stride()[3],
        output.stride()[0], output.stride()[1], output.stride()[2], output.stride()[3],
        eps=1e-05,
        USE_FP32=use_fp32
    )
    return output

def replacement_func():
    return fused_conv_bn_impl