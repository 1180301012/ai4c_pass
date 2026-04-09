import torch
import triton
import triton.language as tl

# Pattern matching function to match just BatchNorm
def pattern(in_4, in_0, in_1, in_3, in_2):
    # BatchNorm: input=in_4, running_mean=in_0, running_var=in_1, weight=in_3, bias=in_2
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_4

def replacement_args(in_4, in_0, in_1, in_3, in_2):
    return (in_4, in_0, in_1, in_3, in_2)

@triton.jit
def simple_batch_norm_kernel(
    x_ptr,
    running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    batch_size, num_channels, height, width,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.cdiv(batch_size * num_channels * height * width, BLOCK_SIZE)
    
    if pid >= num_pid:
        return
    
    # Efficient tile-based processing
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * num_channels * height * width)
    
    # Convert linear offset to 4D coordinates with efficient strides
    b = offsets // (num_channels * height * width)
    c = (offsets % (num_channels * height * width)) // (height * width)
    spatial = offsets % (height * width)
    h = spatial // width
    w = spatial % width
    
    # Load input data with cache-friendly access
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load BatchNorm parameters for each thread with proper bounds checking
    # Use tl.where for conditional loading to avoid mask issues
    channel_indices = tl.broadcast_to(c, (BLOCK_SIZE,))
    mean = tl.where(c < num_channels, tl.load(running_mean_ptr + c), 0.0)
    var = tl.where(c < num_channels, tl.load(running_var_ptr + c), 1.0)
    weight = tl.where(c < num_channels, tl.load(weight_ptr + c), 1.0)
    bias = tl.where(c < num_channels, tl.load(bias_ptr + c), 0.0)
    
    # Apply optimized BatchNorm computation
    # Combine operations to minimize memory access
    var_plus_eps = var + eps
    inv_std = tl.math.rsqrt(var_plus_eps)
    x_norm = (x - mean) * inv_std
    x_bn = x_norm * weight + bias
    
    # Store result efficiently
    tl.store(out_ptr + offsets, x_bn, mask=mask)

@torch.fx.wrap
def simple_batch_norm(in_4, in_0, in_1, in_3, in_2):
    batch_size, num_channels, height, width = in_4.shape
    
    out = torch.empty_like(in_4)
    
    # Optimized block size for cache efficiency
    BLOCK_SIZE = 1024
    
    # Calculate number of programs and launch kernel
    num_programs = (batch_size * num_channels * height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    simple_batch_norm_kernel[(num_programs,)](
        in_4, in_0, in_1, in_3, in_2, out,
        batch_size, num_channels, height, width,
        1e-05, BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return simple_batch_norm