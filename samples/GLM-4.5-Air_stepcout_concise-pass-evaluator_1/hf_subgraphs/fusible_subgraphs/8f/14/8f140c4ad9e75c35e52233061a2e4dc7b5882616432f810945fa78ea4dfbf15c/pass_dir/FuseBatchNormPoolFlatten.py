import torch
import triton
import triton.language as tl

def pattern(in_7, in_0, in_1, in_3, in_2):
    # Batch normalization with the specific signature from the original model
    tmp_8 = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    # Adaptive average pooling to 1x1
    tmp_9 = torch.nn.functional.adaptive_avg_pool2d(tmp_8, 1)
    # Flatten to feature vector
    tmp_10 = tmp_9.flatten(1, -1)
    return tmp_10

def replacement_args(in_7, in_0, in_1, in_3, in_2):
    return (in_7, in_0, in_1, in_3, in_2)

@triton.jit
def batch_norm_pool_flatten_kernel(
    input_ptr,           # [B, C, H, W] - input tensor after conv+add
    running_mean_ptr,    # [C] - batch norm running mean
    running_var_ptr,     # [C] - batch norm running var
    weight_ptr,          # [C] - batch norm weight (scale)
    bias_ptr,            # [C] - batch norm bias
    output_ptr,          # [B, C] - flattened output
    batch_size,          # batch size
    channels,            # number of channels
    height,              # input height
    width,               # input width
    momentum,            # 0.1
    eps,                 # 1e-05
    BLOCK_SIZE_C: tl.constexpr,  # block size for channels
    BLOCK_SIZE_B: tl.constexpr,  # block size for batch
):
    # Each program handles one batch and one channel
    b = tl.program_id(0)
    c = tl.program_id(1)
    
    # Load batch norm parameters for this channel
    mean = tl.load(running_mean_ptr + c, mask=(c < channels), other=0.0)
    var = tl.load(running_var_ptr + c, mask=(c < channels), other=1.0)  # default to 1.0 for safety
    weight = tl.load(weight_ptr + c, mask=(c < channels), other=1.0)
    bias = tl.load(bias_ptr + c, mask=(c < channels), other=0.0)
    
    # Compute scale and inv_std for batch norm
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Initialize accumulator for average pooling
    sum_val = 0.0
    
    # Process all pixels in spatial dimensions (will average them)
    for h in range(height):
        for w in range(width):
            # Calculate input offset
            input_offset = (b * channels + c) * height * width + h * width + w
            
            # Load input value
            x = tl.load(input_ptr + input_offset, mask=(input_offset < batch_size * channels * height * width), other=0.0)
            
            # Apply batch normalization
            # y = (x - mean) * (weight * inv_std) + bias
            norm_x = (x - mean) * inv_std
            batch_norm_x = norm_x * weight + bias
            
            # Accumulate for average pooling
            sum_val += batch_norm_x
    
    # Compute average (pooling to 1x1)
    avg_val = sum_val / (height * width)
    
    # Store result in flattened output
    output_offset = b * channels + c
    tl.store(output_ptr + output_offset, avg_val, mask=(output_offset < batch_size * channels))

@torch.fx.wrap
def fused_batch_norm_pool_flatten(input_tensor, running_mean, running_var, bn_weight, bn_bias):
    B, C, H, W = input_tensor.shape
    
    # Create output tensor (flattened)
    output = torch.empty((B, C), dtype=torch.float32, device=input_tensor.device)
    
    # Launch kernel
    batch_norm_pool_flatten_kernel[(B, C)](
        input_ptr=input_tensor,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=bn_weight,
        bias_ptr=bn_bias,
        output_ptr=output,
        batch_size=B,
        channels=C,
        height=H,
        width=W,
        momentum=0.1,
        eps=1e-05,
        BLOCK_SIZE_C=1,  # Process one channel at a time
        BLOCK_SIZE_B=1,  # Process one batch at a time
    )
    
    return output

def replacement_func():
    return fused_batch_norm_pool_flatten