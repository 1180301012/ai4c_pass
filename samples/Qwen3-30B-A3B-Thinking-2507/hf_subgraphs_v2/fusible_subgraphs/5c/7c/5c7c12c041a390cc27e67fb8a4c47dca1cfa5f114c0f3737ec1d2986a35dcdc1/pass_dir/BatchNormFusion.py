import torch
import triton
import triton.language as tl

# Pattern matching for batch_norm with fixed parameters
# Expected pattern: batch_norm(input, running_mean, running_var, weight, bias, False, 0.1, 0.001)
def pattern(tmp_6, in_0, in_1, in_3, in_2):
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    return tmp_7

# Extract arguments needed for the replacement
# Returns (input, running_mean, running_var, weight, bias)
def replacement_args(tmp_6, in_0, in_1, in_3, in_2):
    return (tmp_6, in_0, in_1, in_3, in_2)

@triton.jit
def batch_norm_kernel(
    in_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    """Custom batch norm kernel for inference (using running stats)"""
    pid = tl.program_id(0)
    num_elements = batch_size * channels * height * width
    start = pid * BLOCK_SIZE
    end = min(start + BLOCK_SIZE, num_elements)

    for i in range(start, end):
        # Convert flat index to 4D indices
        batch_idx = i // (channels * height * width)
        channel_idx = (i % (channels * height * width)) // (height * width)
        h_idx = (i % (height * width)) // width
        w_idx = i % width

        # Load input value
        x = tl.load(in_ptr + i)
        
        # Load running stats for this channel
        mean_val = tl.load(mean_ptr + channel_idx)
        var_val = tl.load(var_ptr + channel_idx)
        weight_val = tl.load(weight_ptr + channel_idx)
        bias_val = tl.load(bias_ptr + channel_idx)

        # Compute: (x - mean) / sqrt(var + eps) * weight + bias
        norm = 1.0 / tl.sqrt(var_val + eps)
        out = (x - mean_val) * norm * weight_val + bias_val
        tl.store(out_ptr + i, out)

@torch.fx.wrap
def batch_norm_triton(input_tensor, running_mean, running_var, weight, bias, eps=0.001):
    batch_size, channels, height, width = input_tensor.shape
    num_elements = batch_size * channels * height * width
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Get raw pointers
    input_ptr = input_tensor.data_ptr()
    mean_ptr = running_mean.data_ptr()
    var_ptr = running_var.data_ptr()
    weight_ptr = weight.data_ptr()
    bias_ptr = bias.data_ptr()
    output_ptr = output.data_ptr()

    # Define block size and grid size
    BLOCK_SIZE = 1024
    grid_size = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel
    batch_norm_kernel[grid_size](
        input_ptr,
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        channels,
        height,
        width,
        eps,
        BLOCK_SIZE
    )

    return output

def replacement_func():
    return batch_norm_triton