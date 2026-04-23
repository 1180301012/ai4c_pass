import torch
import triton
import triton.language as tl

# ============================================================================
# Pattern A: conv2d(in_6, in_4) -> batch_norm -> add in_5
# Returns: (result,)
# ============================================================================
def pattern_a(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Pattern for deeppose_resnet_101: Conv2d + BN + Add
    in_6: input [N, C_in, H, W]
    in_4: conv weight [C_out, C_in, 1, 1]
    in_0: running_mean [C_out]
    in_1: running_var [C_out]
    in_3: bn_weight [C_out]
    in_2: bn_bias [C_out]
    in_5: residual [N, C_out, H, W]
    """
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 += in_5
    return tmp_6


# ============================================================================
# Pattern B: conv2d(in_5, in_0) -> batch_norm -> add in_6 (in-place)
# Returns: (result,)
# ============================================================================
def pattern_b(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Pattern for resnet10t: Conv2d + BN + Add (in-place)
    in_5: input [N, C_in, H, W]
    in_0: conv weight [C_out, C_in, 1, 1]
    in_1: running_mean [C_out]
    in_2: running_var [C_out]
    in_4: bn_weight [C_out]
    in_3: bn_bias [C_out]
    in_6: residual (modified in-place) [N, C_out, H, W]
    """
    conv2d = torch.conv2d(in_5, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    in_6 += tmp_6
    return in_6


# ============================================================================
# Argument extraction functions
# ============================================================================
def replacement_args_a(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, "route_a")


def replacement_args_b(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, "route_b")


# ============================================================================
# Triton Kernels
# ============================================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'num_stages': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'num_stages': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'num_stages': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'num_stages': 2}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'num_stages': 1}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'num_stages': 1}, num_warps=4),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def fused_conv_bn_add_kernel_route_a(
    input_ptr, weight_ptr, running_mean_ptr, running_var_ptr, bn_weight_ptr, bn_bias_ptr,
    residual_ptr, output_ptr,
    N, C, H, W, C_in,
    stride_n, stride_c, stride_h, stride_w,
    BN_MEAN_VAR_EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, num_stages: tl.constexpr
):
    """
    Fused kernel for Pattern A:
    1. Apply 1x1 conv (rearranged as element-wise multiply + sum over channels)
    2. BatchNorm (normalize + scale + shift)
    3. Add residual
    
    Note: 1x1 conv with weight [C_out, C_in, 1, 1] can be implemented as:
    output[n, c_out, h, w] = sum_c_in(input[n, c_in, h, w] * weight[c_out, c_in, 0, 0])
    This is essentially a batched matrix multiply where weight is transposed.
    """
    pid = tl.program_id(axis=0)
    num_pid_n = N
    num_pid_c = C
    num_pid_hw = H * W
    
    num_pid_in_grid = num_pid_n * num_pid_c * num_pid_hw
    pid_hw = pid % num_pid_hw
    pid_c = (pid // num_pid_hw) % num_pid_c
    pid_n = pid // (num_pid_hw * num_pid_c)
    
    h = pid_hw // W
    w = pid_hw % W
    
    # Base offsets
    input_offset = pid_n * stride_n + pid_hw * 1  # This needs proper stride
    # Strides for input: [stride_n, stride_c, stride_h, stride_w]
    
    # Load running mean and var for this channel
    mean = tl.load(running_mean_ptr + pid_c).to(tl.float32)
    var = tl.load(running_var_ptr + pid_c).to(tl.float32)
    gamma = tl.load(bn_weight_ptr + pid_c).to(tl.float32)
    beta = tl.load(bn_bias_ptr + pid_c).to(tl.float32)
    
    # Compute 1/sqrt(var + eps) and normalized scale
    inv_std = 1.0 / tl.sqrt(var + BN_MEAN_VAR_EPS)
    normalized_scale = gamma * inv_std
    normalized_bias = beta - mean * gamma * inv_std
    
    # Compute conv (1x1 = channel-wise weighted sum)
    # For each output pixel, sum over input channels
    conv_result = tl.zeros((1,), dtype=tl.float32)
    
    for c_in_idx in range(0, C_in, BLOCK_SIZE_N):
        a_offset = pid_n * stride_n + c_in_idx * 1  # Wrong, need proper indexing
        
        # Correct offset calculation: offset = n*stride_n + c*stride_c + h*stride_h + w*stride_w
        a_offset = pid_n * stride_n + c_in_idx * stride_c + h * stride_h + w * stride_w
        b_offset = pid_c * C_in * 1 * 1 + c_in_idx * 1 * 1  # weight[c_out, c_in, 0, 0]
        
        mask_a = (pid_n < N) & (c_in_idx < C_in) & (h < H) & (w < W)
        mask_b = (pid_c < C) & (c_in_idx < C_in)
        
        a = tl.load(input_ptr + a_offset, mask=mask_a, other=0.0).to(tl.float32)
        b = tl.load(weight_ptr + b_offset, mask=mask_b, other=0.0).to(tl.float32)
        
        conv_result += a * b
    
    # Apply BN
    bn_result = conv_result * normalized_scale + normalized_bias
    
    # Load residual and add
    residual_offset = pid_n * stride_n + pid_c * stride_c + h * stride_h + w * stride_w
    mask_res = (pid_n < N) & (pid_c < C) & (h < H) & (w < W)
    residual = tl.load(residual_ptr + residual_offset, mask=mask_res, other=0.0).to(tl.float32)
    
    output = bn_result + residual
    
    # Store result
    output_offset = pid_n * stride_n + pid_c * stride_c + h * stride_h + w * stride_w
    tl.store(output_ptr + output_offset, output, mask=mask_res)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'num_stages': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'num_stages': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'num_stages': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'num_stages': 2}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'num_stages': 1}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'num_stages': 1}, num_warps=4),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def fused_conv_bn_add_kernel_route_b(
    input_ptr, weight_ptr, running_mean_ptr, running_var_ptr, bn_weight_ptr, bn_bias_ptr,
    residual_ptr, output_ptr,
    N, C, H, W, C_in,
    stride_n, stride_c, stride_h, stride_w,
    BN_MEAN_VAR_EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, num_stages: tl.constexpr
):
    """
    Fused kernel for Pattern B (resnet10t):
    1. Apply 1x1 conv
    2. BatchNorm
    3. Add to residual (in-place on residual, but we return new output)
    
    Here in_5=input, in_0=weight, residual is in_6
    """
    pid = tl.program_id(axis=0)
    num_pid_n = N
    num_pid_c = C
    num_pid_hw = H * W
    
    pid_hw = pid % num_pid_hw
    pid_c = (pid // num_pid_hw) % num_pid_c
    pid_n = pid // (num_pid_hw * num_pid_c)
    
    h = pid_hw // W
    w = pid_hw % W
    
    # Load BN stats
    mean = tl.load(running_mean_ptr + pid_c).to(tl.float32)
    var = tl.load(running_var_ptr + pid_c).to(tl.float32)
    gamma = tl.load(bn_weight_ptr + pid_c).to(tl.float32)
    beta = tl.load(bn_bias_ptr + pid_c).to(tl.float32)
    
    inv_std = 1.0 / tl.sqrt(var + BN_MEAN_VAR_EPS)
    normalized_scale = gamma * inv_std
    normalized_bias = beta - mean * gamma * inv_std
    
    # Compute 1x1 conv (grouped convolution with 1x1 kernel = per-pixel channel mixing)
    conv_result = tl.zeros((1,), dtype=tl.float32)
    
    for c_in_idx in range(0, C_in, BLOCK_SIZE_N):
        a_offset = pid_n * stride_n + c_in_idx * stride_c + h * stride_h + w * stride_w
        b_offset = pid_c * C_in * 1 * 1 + c_in_idx * 1 * 1  # weight layout
        
        mask_a = (pid_n < N) & (c_in_idx < C_in) & (h < H) & (w < W)
        mask_b = (pid_c < C) & (c_in_idx < C_in)
        
        a = tl.load(input_ptr + a_offset, mask=mask_a, other=0.0).to(tl.float32)
        b = tl.load(weight_ptr + b_offset, mask=mask_b, other=0.0).to(tl.float32)
        
        conv_result += a * b
    
    bn_result = conv_result * normalized_scale + normalized_bias
    
    # Add to residual (residual_ptr = in_6)
    residual_offset = pid_n * stride_n + pid_c * stride_c + h * stride_h + w * stride_w
    mask_res = (pid_n < N) & (pid_c < C) & (h < H) & (w < W)
    residual = tl.load(residual_ptr + residual_offset, mask=mask_res, other=0.0).to(tl.float32)
    
    output = bn_result + residual
    
    output_offset = pid_n * stride_n + pid_c * stride_c + h * stride_h + w * stride_w
    tl.store(output_ptr + output_offset, output, mask=mask_res)


@torch.fx.wrap
def _fused_conv_bn_add_dispatch(
    running_mean, running_var, bn_weight, bn_bias,
    conv_weight, input_tensor, residual, output,
    N, C, H, W, C_in,
    stride_n, stride_c, stride_h, stride_w,
    route
):
    """
    Dispatch wrapper - routes to the appropriate Triton kernel based on route string.
    """
    BN_EPS = 1e-05
    total_elements = N * C * H * W
    
    grid = (total_elements,)
    
    if route == "route_a":
        fused_conv_bn_add_kernel_route_a[grid](
            input_tensor, conv_weight, running_mean, running_var, bn_weight, bn_bias,
            residual, output,
            N, C, H, W, C_in,
            stride_n, stride_c, stride_h, stride_w,
            BN_EPS,
        )
    elif route == "route_b":
        fused_conv_bn_add_kernel_route_b[grid](
            input_tensor, conv_weight, running_mean, running_var, bn_weight, bn_bias,
            residual, output,
            N, C, H, W, C_in,
            stride_n, stride_c, stride_h, stride_w,
            BN_EPS,
        )
    else:
        raise ValueError(f"Unknown route: {route}")
    
    return output


def replacement_func():
    return _fused_conv_bn_add_dispatch