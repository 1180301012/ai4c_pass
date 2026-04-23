"""
Shared Triton kernels for Conv2d(1x1) + BatchNorm(eval) + Residual Add fusion.
All passes route through these kernels via the shared replacement_func dispatch wrapper.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_bn_add_kernel(
    input_ptr, weight_ptr, residual_ptr, output_ptr,
    mean_ptr, var_ptr, bn_weight_ptr, bn_bias_ptr,
    N, C_in, C_out, HW,
    stride_in_n, stride_in_c, stride_in_hw,
    stride_w_co, stride_w_ci,
    stride_res_n, stride_res_c, stride_res_hw,
    stride_out_n, stride_out_c, stride_out_hw,
    eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fused Conv2d(1x1) + BatchNorm(eval) + Residual Add kernel.
    
    A 1x1 conv2d with stride=1, padding=0, dilation=1, groups=1 is a batched matmul:
      output[n, c_out, h, w] = sum_{c_in} weight[c_out, c_in] * input[n, c_in, h, w]
    
    BN in eval mode: bn_out = (conv - mean) / sqrt(var + eps) * weight + bias
                     = conv * scale + offset
    where scale = bn_weight / sqrt(bn_var + eps), offset = bn_bias - bn_mean * scale
    
    Final result = bn_out + residual (addition is commutative)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < C_out
    n_mask = offs_n < HW

    # Load BN parameters for output channels in this tile and precompute scale/offset
    mean_val = tl.load(mean_ptr + offs_m, mask=m_mask, other=0.0).to(tl.float32)
    var_val = tl.load(var_ptr + offs_m, mask=m_mask, other=0.0).to(tl.float32)
    bn_w_val = tl.load(bn_weight_ptr + offs_m, mask=m_mask, other=1.0).to(tl.float32)
    bn_b_val = tl.load(bn_bias_ptr + offs_m, mask=m_mask, other=0.0).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var_val + eps)
    bn_scale = bn_w_val * inv_std
    bn_offset = bn_b_val - mean_val * bn_scale

    # Matmul accumulation for 1x1 conv2d
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, C_in, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < C_in

        # Load weight tile [BLOCK_M, BLOCK_K] from [C_out, C_in] matrix
        w_ptrs = weight_ptr + offs_m[:, None] * stride_w_co + offs_k[None, :] * stride_w_ci
        w_full_mask = m_mask[:, None] & k_mask[None, :]
        w = tl.load(w_ptrs, mask=w_full_mask, other=0.0)

        # Load input tile [BLOCK_K, BLOCK_N] from [N, C_in, HW] tensor
        x_ptrs = input_ptr + pid_batch * stride_in_n + offs_k[:, None] * stride_in_c + offs_n[None, :] * stride_in_hw
        x_full_mask = k_mask[:, None] & n_mask[None, :]
        x = tl.load(x_ptrs, mask=x_full_mask, other=0.0)

        acc += tl.dot(w, x, allow_tf32=True)

    # Apply BN: bn_out = conv_out * bn_scale + bn_offset
    bn_out = acc * bn_scale[:, None] + bn_offset[:, None]

    # Load residual tile [BLOCK_M, BLOCK_N] from [N, C_out, HW] tensor
    res_ptrs = residual_ptr + pid_batch * stride_res_n + offs_m[:, None] * stride_res_c + offs_n[None, :] * stride_res_hw
    res_full_mask = m_mask[:, None] & n_mask[None, :]
    res_val = tl.load(res_ptrs, mask=res_full_mask, other=0.0).to(tl.float32)

    # Add residual (commutative: bn_out + residual = residual + bn_out)
    result = bn_out + res_val

    # Store output
    out_ptrs = output_ptr + pid_batch * stride_out_n + offs_m[:, None] * stride_out_c + offs_n[None, :] * stride_out_hw
    out_full_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, result, mask=out_full_mask)


@torch.fx.wrap
def fused_conv2d_bn_add(
    input_tensor, conv_weight, residual,
    bn_mean, bn_var, bn_weight, bn_bias,
    eps,
):
    """Fused Conv2d(1x1) + BatchNorm(eval) + Residual Add wrapper."""
    # Ensure contiguity for the flattened HW stride assumption
    input_tensor = input_tensor.contiguous()
    conv_weight = conv_weight.contiguous()
    residual = residual.contiguous()

    N, C_in, H, W = input_tensor.shape
    C_out = conv_weight.shape[0]
    HW = H * W

    # Ensure BN params are contiguous 1D tensors
    bn_mean = bn_mean.contiguous()
    bn_var = bn_var.contiguous()
    bn_weight = bn_weight.contiguous()
    bn_bias = bn_bias.contiguous()

    output = torch.empty((N, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)

    # Choose block sizes based on tensor dimensions
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Adjust block sizes for small dimensions
    if C_out < BLOCK_M:
        BLOCK_M = max(16, ((C_out + 15) // 16) * 16)
    if HW < BLOCK_N:
        BLOCK_N = max(16, ((HW + 15) // 16) * 16)
    if C_in < BLOCK_K:
        BLOCK_K = max(16, ((C_in + 15) // 16) * 16)

    grid_m = (C_out + BLOCK_M - 1) // BLOCK_M
    grid_n = (HW + BLOCK_N - 1) // BLOCK_N

    # For contiguous [N, C, H, W] tensors, stride_hw = 1
    stride_in_hw = 1
    stride_res_hw = 1
    stride_out_hw = 1

    conv2d_bn_add_kernel[(grid_m, grid_n, N)](
        input_ptr=input_tensor,
        weight_ptr=conv_weight,
        residual_ptr=residual,
        output_ptr=output,
        mean_ptr=bn_mean,
        var_ptr=bn_var,
        bn_weight_ptr=bn_weight,
        bn_bias_ptr=bn_bias,
        N=N, C_in=C_in, C_out=C_out, HW=HW,
        stride_in_n=input_tensor.stride()[0],
        stride_in_c=input_tensor.stride()[1],
        stride_in_hw=stride_in_hw,
        stride_w_co=conv_weight.stride()[0],
        stride_w_ci=conv_weight.stride()[1],
        stride_res_n=residual.stride()[0],
        stride_res_c=residual.stride()[1],
        stride_res_hw=stride_res_hw,
        stride_out_n=output.stride()[0],
        stride_out_c=output.stride()[1],
        stride_out_hw=stride_out_hw,
        eps=eps,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return output