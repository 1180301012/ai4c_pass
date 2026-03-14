import torch
import triton
import triton.language as tl

@triton.jit
def fused_bn_relu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    n_elements,
    num_features,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    training: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load normalization parameters (each channel has same value across all spatial locations and batches)
    channel_idx = offsets % num_features
    running_mean = tl.load(running_mean_ptr + channel_idx, mask=channel_idx < num_features, other=0.0)
    running_var = tl.load(running_var_ptr + channel_idx, mask=channel_idx < num_features, other=1.0)
    weight = tl.load(weight_ptr + channel_idx, mask=channel_idx < num_features, other=1.0)
    bias = tl.load(bias_ptr + channel_idx, mask=channel_idx < num_features, other=0.0)
    
    # BatchNorm computation
    # Normalize: (x - running_mean) / sqrt(running_var + eps)
    variance = running_var + eps
    inv_std = tl.sqrt(tl.maximum(variance, 1e-5))
    normalized = (x - running_mean) * inv_std
    # Scale and shift: weight * normalized + bias
    bn_out = weight * normalized + bias
    
    # ReLU activation
    out = tl.maximum(bn_out, 0.0)
    
    # Store result
    tl.store(y_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_batch_norm_relu(x, running_mean, running_var, weight, bias, training=True, momentum=0.1, eps=1e-5):
    # Determine tensor shapes
    N, C, H, W = x.shape
    n_elements = N * C * H * W
    
    # Choose block size based on tensor size
    BLOCK_SIZE = 1024
    if n_elements < 1024:
        BLOCK_SIZE = 128
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    y = torch.empty_like(x)
    
    # Launch kernel
    fused_bn_relu_kernel[(num_programs,)](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        y_ptr=y,
        n_elements=n_elements,
        num_features=C,
        eps=eps,
        momentum=momentum,
        training=training,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y

def pattern(input, running_mean, running_var, weight, bias, training, momentum, eps):
    # Match the exact computation from the model
    # tmp_9 = torch.nn.functional.batch_norm(tmp_8, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 1e-05)
    bn_out = torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
    # tmp_10 = torch.nn.functional.relu(tmp_9, inplace=True)
    relu_out = torch.nn.functional.relu(bn_out, inplace=True)
    # The model returns (tmp_10,), so we return relu_out to match
    return relu_out

def replacement_args(x, running_mean, running_var, weight, bias, training, momentum, eps):
    return (x, running_mean, running_var, weight, bias, training, momentum, eps)

def replacement_func():
    return fused_batch_norm_relu