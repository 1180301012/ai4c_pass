import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv_bn_relu_kernel(
    input_ptr, weight_ptr, running_mean_ptr, running_var_ptr, 
    weight_bn_ptr, bias_bn_ptr, 
    output_ptr,
    n_elements,
    # Tensor dimensions
    n_channels: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    kernel_size: tl.constexpr,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Depthwise Conv2d + BatchNorm + ReLU kernel.
    
    For depthwise convolution (groups=C), we can fuse batch norm by computing
    effective_bias = bias_bn - weight_bn * running_mean / sqrt(running_var + eps)
    Then: output = conv(input, weight) + effective_bias, clamped at 0
    """
    pid = tl.program_id(0)
    
    # Each program processes BLOCK_SIZE elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate channel, height, width indices
    # Layout: [C, H, W] for each element
    c_idx = offsets // (H * W)
    h_idx = (offsets % (H * W)) // W
    w_idx = offsets % W
    
    # Load running mean and variance for this channel
    running_mean = tl.load(running_mean_ptr + c_idx, mask=mask)
    running_var = tl.load(running_var_ptr + c_idx, mask=mask)
    gamma = tl.load(weight_bn_ptr + c_idx, mask=mask)
    beta = tl.load(bias_bn_ptr + c_idx, mask=mask)
    
    # Compute effective bias: beta - gamma * mean / sqrt(var + eps)
    eps = 1e-05
    inv_std = 1.0 / tl.sqrt(running_var + eps)
    effective_bias = beta - gamma * running_mean * inv_std
    
    # Compute convolution output
    # For depthwise conv with 7x7 kernel, stride=1, padding=3 (to maintain spatial size 70->64)
    # Wait, let me check the dimensions: input is 70x70, output is 64x64
    # With stride=1 and no padding, 70 - 7 + 1 = 64 ✓
    # Actually with default stride of 1 and no padding, the output would be 70 - 7 + 1 = 64
    
    conv_out = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Kernel loop
    kh_max = kernel_size
    kw_max = kernel_size
    pad_h = kernel_size // 2  # Assuming same padding
    pad_w = kernel_size // 2
    
    for kh in range(kh_max):
        for kw in range(kw_max):
            # Input coordinates with padding
            in_h = h_idx + kh - pad_h
            in_w = w_idx + kw - pad_w
            
            # Check bounds
            in_mask = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)
            in_idx = c_idx * H * W + in_h * W + in_w
            
            # Load input and weight
            inp = tl.load(input_ptr + in_idx, mask=mask & in_mask, other=0.0)
            wt = tl.load(weight_ptr + c_idx * kernel_size * kernel_size + kh * kernel_size + kw, mask=mask)
            
            conv_out = conv_out + inp * wt
    
    # Add effective bias and apply ReLU
    output = tl.maximum(conv_out + effective_bias, 0.0)
    
    # Store output
    tl.store(output_ptr + offsets, output, mask=mask)


def pattern(in_0, in_1, in_2, in_3, in_4, in_9):
    """
    Match: conv2d -> view -> batch_norm -> relu
    """
    # Conv2d with groups=512
    # Note: Using positional args as in model.py: torch.conv2d(input, weight, groups=512)
    tmp_0 = torch.conv2d(in_9, in_4, 512)
    
    # View/reshape
    tmp_1 = tmp_0.view(1, 512, 64, 64)
    
    # Batch norm with running mean, var, weight, bias
    # model.py: batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
    tmp_2 = torch.nn.functional.batch_norm(tmp_1, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    
    # ReLU
    tmp_3 = torch.nn.functional.relu(tmp_2, inplace=False)
    
    return tmp_0, tmp_1, tmp_2, tmp_3


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_9):
    """
    Extract arguments for the fused kernel.
    in_0: running_mean [512]
    in_1: running_var [512]
    in_2: bias [512]
    in_3: weight/gamma [512]
    in_4: conv weight [512, 1, 7, 7]
    in_9: input [1, 512, 70, 70]
    """
    return (in_0, in_1, in_2, in_3, in_4, in_9)


@torch.fx.wrap
def fused_conv_bn_relu_wrapper(in_0, in_1, in_2, in_3, in_4, in_9):
    """
    Wrapper for the fused Conv2d + BatchNorm + ReLU kernel.
    """
    # Input: [1, 512, 70, 70]
    # Output: [1, 512, 64, 64]
    N, C, H, W = in_9.shape
    kernel_size = 7
    
    # Flatten for processing
    n_elements = C * (H - kernel_size + 1) * (W - kernel_size + 1)
    out_H = H - kernel_size + 1  # 70 - 7 + 1 = 64
    out_W = W - kernel_size + 1  # 64
    
    output = torch.empty(1, C, out_H, out_W, dtype=in_9.dtype, device=in_9.device)
    
    # For bfloat16/float16, use float32 for accumulation
    input_ptr = in_9
    weight_ptr = in_4
    running_mean_ptr = in_0
    running_var_ptr = in_1
    weight_bn_ptr = in_3
    bias_bn_ptr = in_2
    
    BLOCK_SIZE = 1024
    num_programs = (C * out_H * out_W + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_conv_bn_relu_kernel[(num_programs,)](
        input_ptr, weight_ptr, running_mean_ptr, running_var_ptr,
        weight_bn_ptr, bias_bn_ptr,
        output,
        n_elements,
        C, out_H, out_W, kernel_size,
        BLOCK_SIZE
    )
    
    return output


def replacement_func():
    return fused_conv_bn_relu_wrapper