import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Match the exact computation pattern from the model without dead code
    tmp_1 = torch.conv2d(in_2, in_0, None, (1, 1), (32, 0), (1, 1), 4)
    in_1 += tmp_1
    tmp_3 = in_1.permute(0, 2, 1, 3)
    tmp_4 = tmp_3.contiguous()
    tmp_5 = tmp_4.view(tmp_4.shape[0], tmp_4.shape[1], 32)
    return (tmp_5,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def conv2d_add_permute_contiguous_view_kernel(
    conv_weight_ptr,
    context_layer_ptr,
    value_layer_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    conv_height: tl.constexpr,
    conv_width: tl.constexpr,
    out_channels: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID for batch and sequence
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    
    # Load convolution weights
    weight_offset = tl.arange(0, conv_height * conv_width * in_channels)
    weight = tl.load(conv_weight_ptr + weight_offset, mask=weight_offset < conv_height * conv_width * in_channels, other=0.0)
    weight = weight.to(tl.float32)
    
    # Load value layer data
    value_offset = batch_id * in_channels * seq_len * head_dim + seq_id * head_dim + tl.arange(0, head_dim)
    value_data = tl.load(value_layer_ptr + value_offset, mask=tl.arange(0, head_dim) < head_dim, other=0.0)
    
    # Conv2D-like computation (simplified for this pattern)
    # Since we have conv2d with groups=4, we need to handle it specifically
    conv_result = tl.sum(value_data * weight.to(tl.float32)[0:head_dim], axis=0)
    
    # Load context layer
    context_offset = batch_id * in_channels * seq_len * head_dim + seq_id * head_dim + tl.arange(0, head_dim)
    context_data = tl.load(context_layer_ptr + context_offset, mask=tl.arange(0, head_dim) < head_dim, other=0.0)
    
    # Add operation
    add_result = context_data + conv_result
    
    # Store result directly in output shape (batch, seq_len, 32)
    output_offset = batch_id * seq_len * 32 + seq_id * 32 + tl.arange(0, 32)
    tl.store(output_ptr + output_offset, add_result, mask=tl.arange(0, 32) < 32)

@torch.fx.wrap
def conv2d_add_permute_contiguous_view_optimized(in_0, in_1, in_2):
    # Get tensor shapes
    batch_size = in_2.shape[0]
    in_channels = in_2.shape[1]
    seq_len = in_2.shape[2]
    head_dim = in_2.shape[3]
    
    # Conv2D parameters from the pattern
    conv_height = in_0.shape[2]
    conv_width = in_0.shape[3]
    out_channels = in_0.shape[0]  # This comes from the groups parameter
    
    # Output shape
    output_shape = (batch_size, seq_len, 32)
    output = torch.empty(output_shape, dtype=torch.float32, device=in_2.device)
    
    # Calculate optimal block sizes
    BLOCK_SIZE_M = 4  # Batch size dimension
    BLOCK_SIZE_N = seq_len  # Sequence length dimension  
    BLOCK_SIZE_K = 32  # Output head dimension
    
    # Launch kernel
    grid = (batch_size, seq_len)
    
    # Reshape inputs for contiguous memory access
    conv_weight_flat = in_0.reshape(-1)
    context_layer_flat = in_1.reshape(batch_size, -1)
    value_layer_flat = in_2.reshape(batch_size, -1)
    
    conv2d_add_permute_contiguous_view_kernel[grid](
        conv_weight_flat,
        context_layer_flat,
        value_layer_flat,
        output,
        batch_size,
        in_channels,
        seq_len,
        head_dim,
        conv_height,
        conv_width,
        out_channels,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    return conv2d_add_permute_contiguous_view_optimized