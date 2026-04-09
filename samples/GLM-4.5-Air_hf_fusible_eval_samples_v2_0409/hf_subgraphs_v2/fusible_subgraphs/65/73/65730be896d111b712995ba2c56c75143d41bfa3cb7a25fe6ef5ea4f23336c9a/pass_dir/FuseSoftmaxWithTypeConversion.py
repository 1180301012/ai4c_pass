import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    in_1 += in_0
    in_2 = in_1
    in_1 = in_0 = None
    tmp_1 = in_2.float()
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_1 = None
    tmp_3 = tmp_2.type_as(in_2)
    tmp_2 = in_2 = None
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    tmp_3 = None
    return (tmp_4,)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized softmax kernel for float16/bfloat16
@triton.jit
def softmax_kernel_fused(
    input_ptr,
    output_ptr,
    batch_size,
    n_channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch x channel combination
    batch_channel_idx = tl.program_id(0)
    batch_idx = batch_channel_idx // n_channels
    channel_idx = batch_channel_idx % n_channels
    
    # Position within the spatial dimensions (height * width flattened)
    block_start = tl.program_id(1) * BLOCK_SIZE
    spatial_offset = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = spatial_offset < spatial_size
    
    # Calculate base offset for this batch-channel combination
    # Strides: [batch_stride, channel_stride, height_stride, width_stride]
    base_offset = batch_idx * input_ptr.stride(0) + channel_idx * input_ptr.stride(1)
    
    # Load the spatial data for this batch-channel
    ptr_base = input_ptr + base_offset
    input_spatial = tl.load(ptr_base + spatial_offset, mask=mask, other=-float('inf'))
    
    # Numerically stable softmax
    max_val = tl.max(input_spatial)
    shifted = input_spatial - max_val
    exp_vals = tl.exp(shifted)
    sum_exp = tl.sum(exp_vals)
    softmax_out = exp_vals / sum_exp
    
    # Scale by dropout probability (0.9 since p=0.1)
    dropout_out = softmax_out * 0.9
    
    # Store the result
    tl.store(output_ptr + base_offset + spatial_offset, dropout_out, mask=mask)

@torch.fx.wrap
def fused_softmax_dropout_wrapper(x):
    # Handle different tensor shapes and dtypes
    if x.dim() == 4:  # [batch, channels, height, width]
        batch_size = x.shape[0]
        n_channels = x.shape[1]
        spatial_size = x.shape[2] * x.shape[3]  # height * width
    else:
        # Handle other tensor dimensions if needed
        batch_size = 1
        n_channels = 1
        spatial_size = x.numel()
    
    BLOCK_SIZE = 1024
    num_blocks = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_batch_channels = batch_size * n_channels
    
    # Create output with same dtype and shape
    output = torch.empty_like(x)
    
    # Special handling for float32 (no conversion needed)
    if x.dtype == torch.float32:
        # For float32, use standard approach but still fuse with dropout
        tmp = torch.nn.functional.softmax(x, dim=-1)
        output = torch.nn.functional.dropout(tmp, p=0.1, training=False)
    else:
        # For float16/bfloat16, use fused kernel
        softmax_kernel_fused[(total_batch_channels, num_blocks)](
            x,
            output,
            batch_size,
            n_channels,
            spatial_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    def optimized_forward(in_0, in_1):
        # Step 1: Addition (in-place)
        in_1 += in_0
        
        # Step 2: Fused softmax + dropout with type conversion optimization
        result = fused_softmax_dropout_wrapper(in_1)
        
        return (result,)
    
    return optimized_forward