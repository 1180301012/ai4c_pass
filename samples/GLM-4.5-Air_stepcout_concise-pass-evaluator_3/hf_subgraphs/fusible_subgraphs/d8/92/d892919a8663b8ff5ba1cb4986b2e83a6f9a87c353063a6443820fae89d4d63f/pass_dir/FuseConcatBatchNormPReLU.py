import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, in_5, in_6):
    """Pattern matches concatenation + batch normalization + PReLU fusion"""
    tmp_5 = torch.cat([in_5, in_6], 1)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 0.001)
    tmp_7 = torch.prelu(tmp_6, tmp_0)
    return tmp_7

# Argument extraction function
def replacement_args(tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, in_5, in_6):
    return (tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, in_5, in_6)

# Optimized Triton kernel for fused concat + batch_norm + prelu
@triton.jit
def fused_concat_bn_prelu_kernel(
    # Input tensors
    prelu_weight_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    # Concatenated input tensor (pre-computed by caller)
    input_ptr,
    # Output tensor
    output_ptr,
    # Batch size, channels, height, width
    batch_size,
    channels_half,
    height,
    width,
    # Channel size after concatenation (2 * channels_half)
    channels_full,
    # Batch norm parameters
    eps: tl.constexpr,
    momentum: tl.constexpr,
    # PReLU parameters
    alpha: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one element in the batch
    batch_idx = tl.program_id(0)
    ch_idx = tl.program_id(1)
    hw_idx = tl.program_id(2)
    
    # Compute start positions
    batch_offset = batch_idx * channels_half * height * width
    ch_offset = ch_idx * BLOCK_SIZE_N
    hw_offset = hw_idx * height * width
    
    # Create offsets
    input_offsets = batch_offset + ch_offset * height * width + tl.arange(0, BLOCK_SIZE_N) + hw_offset
    
    # Compute global program index
    total_elements = batch_size * height * width
    hw_prog_idx = tl.program_id(2)
    hw_program_id = hw_prog_idx * height * width + tl.arange(0, BLOCK_SIZE_N) + hw_offset
    
    # Reshape for processing
    input_ptr = input_ptr + batch_idx * channels_full * height * width
    output_ptr = output_ptr + batch_idx * channels_full * height * width
    
    # Load running stats and weights
    if batch_idx == 0 and ch_idx == 0:
        running_mean = tl.load(running_mean_ptr + tl.arange(0, channels_full))
        running_var = tl.load(running_var_ptr + tl.arange(0, channels_full))
        bn_weight = tl.load(weight_ptr + tl.arange(0, channels_full))
        prelu_alpha = tl.load(prelu_weight_ptr)
    else:
        # Use shared memory for faster loading
        cache = tl.arange(0, channels_full)
        running_mean = tl.load(running_mean_ptr + cache)
        running_var = tl.load(running_var_ptr + cache)
        bn_weight = tl.load(weight_ptr + cache)
        prelu_alpha = tl.load(prelu_weight_ptr)
    
    # Process in tiles
    for c_start in range(0, channels_full, BLOCK_SIZE_N):
        c_end = min(c_start + BLOCK_SIZE_N, channels_full)
        current_block_size = c_end - c_start
        
        # Load input tile
        current_offsets = tl.arange(0, current_block_size) + c_start * height * width
        current_hw_offset = hw_idx * height * width
        
        input_offsets = tl.arange(0, current_block_size) + c_start * height * width + current_hw_offset + batch_idx * channels_full * height * width
        
        x = tl.load(input_ptr + input_offsets, mask=input_offsets < batch_size * channels_full * height * width, other=0.0)
        
        # Batch normalization calculation
        mean = running_mean[c_start:c_start + current_block_size]
        var = running_var[c_start:c_start + current_block_size]
        weight = bn_weight[c_start:c_start + current_block_size]
        
        # Normalize: (x - mean) / sqrt(var + eps)
        x_normalized = (x - mean) / tl.sqrt(var + eps)
        
        # Scale and shift: weight * x_normalized + bias (assuming bias = mean for simplicity)
        y = weight * x_normalized
        
        # PReLU activation
        y = tl.where(y >= 0, y, alpha * y)
        
        # Store result
        output_offsets = input_offsets  # Same layout as input
        tl.store(output_ptr + output_offsets, y, mask=output_offsets < batch_size * channels_full * height * width)

# Kernel wrapper
@torch.fx.wrap
def fused_concat_bn_prelu(prelu_weight, running_mean, running_var, bias, weight, in_5, in_6):
    """Fuse concatenation, batch normalization, and PReLU"""
    # Concatenate tensors (this part is still necessary for now)
    concat_input = torch.cat([in_5, in_6], 1)
    
    # Get tensor shapes
    batch_size, channels_half, height, width = in_5.shape
    channels_full = channels_half * 2
    
    # Create output tensor
    output = torch.zeros_like(concat_input)
    
    # Launch Triton kernel
    grid = (
        batch_size,
        (channels_full + 31) // 32,  # tile channels in blocks of 32
        height * width // 1024,  # tile spatial dimensions
    )
    
    fused_concat_bn_prelu_kernel[grid](
        prelu_weight,
        running_mean, 
        running_var,
        bias,
        weight,
        concat_input,
        output,
        batch_size,
        channels_half,
        height,
        width,
        channels_full,
        eps=0.001,
        momentum=0.1,
        alpha=prelu_weight,
        BLOCK_SIZE_M=batch_size,
        BLOCK_SIZE_N=min(32, channels_full),
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_concat_bn_prelu