import torch
import triton
import triton.language as tl

def pattern(in_5, in_1, in_2, in_4, in_3):
    # Match the adaptive_avg_pool2d + batch_norm + relu pattern exactly
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(in_5, (1, 1))
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace = True)
    return tmp_6, tmp_8

def replacement_args(in_5, in_1, in_2, in_4, in_3):
    return (in_5, in_1, in_2, in_4, in_3)



@triton.jit
def simple_pool_bn_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    mean_out_ptr,
    relu_out_ptr,
    n_batch, n_channels, height, width,
    EPS: tl.constexpr
):
    pid = tl.program_id(0)
    
    if pid >= n_batch * n_channels:
        return
    
    # Compute spatial mean for this batch and channel
    spatial_sum = 0.0
    spatial_count = 0
    
    # Sum over spatial dimensions
    for h in range(height):
        for w in range(width):
            x_offset = (pid * height * width) + (h * width + w)
            x_val = tl.load(x_ptr + x_offset)
            spatial_sum += float(x_val)
            spatial_count += 1
    
    # Compute mean
    spatial_mean = spatial_sum / spatial_count
    
    # Load BN parameters
    mean_val = tl.load(running_mean_ptr + (pid % n_channels))
    var_val = tl.load(running_var_ptr + (pid % n_channels))
    weight_val = tl.load(weight_ptr + (pid % n_channels))
    bias_val = tl.load(bias_ptr + (pid % n_channels))
    
    # Apply batch normalization: (x - mean) / sqrt(var + eps) * weight + bias
    denom = tl.sqrt(var_val + EPS)
    normalized = spatial_mean - mean_val
    normalized = normalized / denom
    bn_output = normalized * weight_val + bias_val
    
    # Apply ReLU
    relu_output = bn_output if bn_output > 0 else 0.0
    
    # Store results
    tl.store(mean_out_ptr + pid, spatial_mean)
    tl.store(relu_out_ptr + pid, relu_output)

@torch.fx.wrap
def simple_fused_pool_bn_relu(in_5, in_1, in_2, in_4, in_3):
    # Rename parameters for clarity
    x, running_mean, running_var, weight, bias = in_5, in_1, in_2, in_4, in_3
    
    batch, channels, height, width = x.shape
    
    # Create output tensors in the required format
    means = torch.empty(batch, channels, device=x.device, dtype=x.dtype)
    relu_out = torch.empty(batch, channels, device=x.device, dtype=x.dtype)
    
    # Flatten tensors for kernel
    x_flat = x.reshape(batch * channels, height * width)
    means_flat = means.reshape(-1)
    relu_flat = relu_out.reshape(-1)
    
    EPS = 1e-05
    
    grid_size = batch * channels
    
    simple_pool_bn_kernel[grid_size](
        x_ptr=x_flat,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        mean_out_ptr=means_flat,
        relu_out_ptr=relu_flat,
        n_batch=batch,
        n_channels=channels,
        height=height,
        width=width,
        EPS=EPS
    )
    
    return means.reshape(batch, channels, 1, 1), relu_out.reshape(batch, channels, 1, 1)

def replacement_func():
    return simple_fused_pool_bn_relu