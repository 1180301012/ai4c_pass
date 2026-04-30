import torch
import triton
import triton.language as tl


@triton.jit
def fused_relu_batch_norm_kernel(
    input_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    output_ptr,
    n_elements,
    n_channels,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: ReLU -> BatchNorm -> Dropout(p=0.0)
    
    Optimizations:
    1. Fuses ReLU + BatchNorm into single kernel (reduces memory bandwidth)
    2. Dropout with p=0.0 is a no-op, so we skip it entirely
    
    BatchNorm formula: y = (x - mean) / sqrt(var + eps) * weight + bias
    """
    # Get program ID for channel dimension
    channel_id = tl.program_id(0)
    
    # Calculate statistics for this channel
    mean = tl.load(mean_ptr + channel_id)
    var = tl.load(var_ptr + channel_id)
    weight = tl.load(weight_ptr + channel_id)
    bias = tl.load(bias_ptr + channel_id)
    
    # Compute normalized scale
    inv_std = 1.0 / tl.sqrt(var + eps)
    scale = weight * inv_std
    
    # Process all elements in this channel
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for batch_start in range(0, n_elements, BLOCK_SIZE):
        # Calculate indices
        mask = batch_start + offsets < n_elements
        indices = batch_start + offsets
        
        # Global indices: batch_idx * n_channels + channel_id
        global_indices = indices * n_channels + channel_id
        
        # Load input
        x = tl.load(input_ptr + global_indices, mask=mask, other=0.0)
        
        # Apply ReLU
        x = tl.maximum(x, 0.0)
        
        # Apply BatchNorm: y = (x - mean) * scale + bias
        y = (x - mean) * scale + bias
        
        # Store result (dropout p=0.0 is a no-op, so just store y)
        tl.store(output_ptr + global_indices, y, mask=mask)


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match pattern: ReLU -> BatchNorm -> Dropout(p=0.0)
    
    Args:
    - in_0: running mean for batch norm
    - in_1: running var for batch norm
    - in_2: bias for batch norm
    - in_3: weight for batch norm
    - in_4: input tensor
    """
    tmp_4 = torch.nn.functional.relu(in_4, inplace=False)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.dropout(tmp_5, p=0.0, training=False)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """
    Extract arguments needed for the fused kernel.
    
    The kernel expects:
    - in_0: running mean
    - in_1: running var  
    - in_2: bias
    - in_3: weight
    - in_4: input tensor
    """
    return (in_0, in_1, in_2, in_3, in_4)


@torch.fx.wrap
def fused_relu_batch_norm(in_0, in_1, in_2, in_3, in_4):
    """
    Wrapper function that launches the fused ReLU + BatchNorm kernel.
    
    Input shape: [batch_size, n_channels] where n_channels = 128
    """
    batch_size, n_channels = in_4.shape
    
    # Allocate output tensor
    output = torch.empty_like(in_4)
    
    # Ensure tensors are contiguous for better memory access
    if not in_4.is_contiguous():
        in_4 = in_4.contiguous()
    
    # Launch kernel: one program per channel (n_channels programs)
    BLOCK_SIZE = 1024
    grid = (n_channels,)
    
    fused_relu_batch_norm_kernel[grid](
        in_4, in_0, in_1, in_3, in_2,
        output,
        batch_size,
        n_channels,
        1e-05,
        BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    """
    Return the replacement function for the fused kernel.
    """
    return fused_relu_batch_norm