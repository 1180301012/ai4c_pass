import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """
    Match the SE block pattern:
    1. conv2d: tmp_6 = torch.conv2d(in_7, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
    2. sigmoid: tmp_7 = tmp_6.sigmoid()
    3. multiply: tmp_8 = in_6 * tmp_7
    4. relu: tmp_9 = torch.nn.functional.relu(tmp_8, inplace=True)
    5. batch_norm: tmp_10 = torch.nn.functional.batch_norm(tmp_9, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    
    Returns both tmp_9 (relu output) and tmp_10 (batch_norm output) as required.
    """
    # 1x1 conv2d: in_7 (x_se) with weight (in_5) and bias (in_4)
    tmp_6 = torch.conv2d(in_7, in_5, in_4, (1, 1), (0, 0), (1, 1), 1)
    # Sigmoid activation
    tmp_7 = tmp_6.sigmoid()
    # Multiply main input (in_6/x) with SE output (tmp_7)
    tmp_8 = in_6 * tmp_7
    # ReLU activation (inplace)
    tmp_9 = torch.nn.functional.relu(tmp_8, inplace=True)
    # Batch normalization (training=False, momentum=0.1, eps=1e-05)
    tmp_10 = torch.nn.functional.batch_norm(tmp_9, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_9, tmp_10


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """
    Extract the arguments needed for the fused kernel:
    - in_0: running mean for batch_norm
    - in_1: running var for batch_norm
    - in_2: bias for batch_norm
    - in_3: weight for batch_norm
    - in_4: bias for conv (fc2)
    - in_5: weight for conv (fc2)
    - in_6: main feature input (x)
    - in_7: SE feature input (x_se)
    """
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7)


@triton.autotune(
    configs=[
        # Different tile sizes for different workload sizes
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=5, num_warps=2),
    ],
    key=['N', 'M'],
)
@triton.jit
def fused_se_conv_sigmoid_mul_relu_kernel(
    # Input pointers
    x_ptr, x_se_ptr,  # Main input and SE input
    conv_weight_ptr, conv_bias_ptr,  # Conv weight and bias
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,  # Batch norm params
    # Output pointers
    relu_out_ptr, bn_out_ptr,
    # Sizes
    M, N,  # Output dimensions (M = batch * h * w, N = channels)
    C_se,  # SE channels
    # Strides
    stride_x, stride_x_se, stride_conv_w,
    # Constants
    eps: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused SE block kernel that performs:
    1. Conv2d (1x1): [B, C_se, 1, 1] -> [B, C, 1, 1]
    2. Sigmoid: activation
    3. Multiply: [B, C, H, W] * [B, C, 1, 1] (broadcast)
    4. ReLU: activation
    5. BatchNorm is done separately (not fused due to state)
    """
    # Get program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, M - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Calculate offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Load main input x [B, C, H, W] - load the full channel slice
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x + offs_n[None, :] * 1)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Load SE input x_se [B, C_se, 1, 1] - only need the first element per batch
    # For SE, we have [B, C_se, 1, 1], we need to compute conv to get [B, C, 1, 1]
    # Then apply sigmoid and multiply with broadcast
    
    # Compute 1x1 conv: x_se @ conv_weight + conv_bias
    # x_se is [B, C_se, 1, 1], conv_weight is [C, C_se, 1, 1]
    # Result is [B, C, 1, 1]
    
    # Initialize accumulator for conv output
    conv_out = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Convolution: for each output channel n, sum over SE channels
    for c_se in range(C_se):
        # Load SE input for this channel
        x_se_offs = offs_m * C_se + c_se
        x_se_mask = offs_m < M
        x_se = tl.load(x_se_ptr + x_se_offs, mask=x_se_mask, other=0.0)
        
        # Load conv weight [C, C_se] - flattened
        weight_ptrs = conv_weight_ptr + (offs_n * C_se + c_se)
        weight = tl.load(weight_ptrs, mask=mask_n, other=0.0)
        
        # Accumulate: x_se * weight
        conv_out += x_se[:, None] * weight[None, :]
    
    # Add bias
    bn_bias = tl.load(bn_bias_ptr + offs_n, mask=mask_n, other=0.0)
    conv_out = conv_out + bn_bias[None, :]
    
    # Sigmoid activation: 1 / (1 + exp(-x))
    sigmoid_out = 1.0 / (1.0 + tl.exp(-conv_out))
    
    # Multiply: x * sigmoid_out (broadcast along H,W dims)
    mul_out = x * sigmoid_out
    
    # ReLU activation: max(0, x)
    relu_out = tl.where(mul_out > 0, mul_out, 0.0)
    
    # Store relu output
    tl.store(relu_out_ptr + (offs_m[:, None] * M + offs_n[None, :]), relu_out, mask=mask)
    
    # BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    # Note: Using running mean/var (training=False)
    bn_mean = tl.load(bn_mean_ptr + offs_n, mask=mask_n, other=0.0)
    bn_var = tl.load(bn_var_ptr + offs_n, mask=mask_n, other=0.0)
    bn_weight = tl.load(bn_weight_ptr + offs_n, mask=mask_n, other=0.0)
    # bn_bias already loaded above
    
    # Normalize
    bn_out = (relu_out - bn_mean[None, :]) / tl.sqrt(bn_var[None, :] + eps)
    # Scale and shift
    bn_out = bn_out * bn_weight[None, :] + bn_bias[None, :]
    
    # Store batch norm output
    tl.store(bn_out_ptr + (offs_m[:, None] * M + offs_n[None, :]), bn_out, mask=mask)


def fused_se_kernel(x, x_se, conv_weight, conv_bias, bn_mean, bn_var, bn_weight, bn_bias, eps=1e-05, momentum=0.1):
    """
    Fused SE block kernel that performs:
    1. Conv2d (1x1): [B, C_se, 1, 1] -> [B, C, 1, 1]
    2. Sigmoid: activation
    3. Multiply: [B, C, H, W] * [B, C, 1, 1] (broadcast)
    4. ReLU: activation
    5. BatchNorm: normalization using running stats
    """
    # Get input shapes
    # x: [B, C, H, W]
    # x_se: [B, C_se, 1, 1]
    # conv_weight: [C, C_se, 1, 1]
    # conv_bias: [C]
    # bn_mean, bn_var, bn_weight, bn_bias: [C]
    
    B, C, H, W = x.shape
    C_se = x_se.shape[1]
    
    # Flatten spatial dimensions: [B, C, H, W] -> [B*H*W, C]
    M = B * H * W  # Total spatial positions
    N = C          # Number of channels
    
    # Reshape for kernel
    x_flat = x.permute(0, 2, 3, 1).reshape(M, N)  # [M, N]
    x_se_flat = x_se.squeeze(-1).squeeze(-1)  # [B, C_se]
    
    # Conv weight: [C, C_se, 1, 1] -> [C, C_se]
    conv_weight_flat = conv_weight.squeeze(-1).squeeze(-1)
    if conv_bias is None:
        conv_bias = torch.zeros(C, device=x.device, dtype=x.dtype)
    
    # Allocate outputs
    relu_out_flat = torch.empty((M, N), device=x.device, dtype=x.dtype)
    bn_out_flat = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Launch kernel
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    grid = (M * N // (BLOCK_SIZE_M * BLOCK_SIZE_N) + 1,)
    
    fused_se_conv_sigmoid_mul_relu_kernel[grid](
        x_flat, x_se_flat,
        conv_weight_flat, conv_bias,
        bn_mean, bn_var, bn_weight, bn_bias,
        relu_out_flat, bn_out_flat,
        M, N, C_se,
        x.stride(0), x_se.stride(0), conv_weight_flat.stride(0),
        eps, momentum,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    # Reshape outputs back to [B, C, H, W]
    relu_out = relu_out_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
    bn_out = bn_out_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
    
    return relu_out, bn_out


@torch.fx.wrap
def fused_se_kernel_wrapper(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """
    Wrapper function that matches the signature expected by the pattern replacement.
    
    Args:
    - in_0: running mean (bn_mean)
    - in_1: running var (bn_var) 
    - in_2: bias (bn_bias)
    - in_3: weight (bn_weight)
    - in_4: conv bias (fc2_bias)
    - in_5: conv weight (fc2_weight)
    - in_6: main input (x)
    - in_7: SE input (x_se)
    """
    return fused_se_kernel(
        in_6,  # x - main input
        in_7,  # x_se - SE input
        in_5,  # conv_weight
        in_4,  # conv_bias
        in_0,  # bn_mean
        in_1,  # bn_var
        in_3,  # bn_weight
        in_2,  # bn_bias
    )


def replacement_func():
    """
    Return the replacement function.
    """
    return fused_se_kernel_wrapper