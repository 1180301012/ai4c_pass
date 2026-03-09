import torch
import triton
import triton.language as tl

def pattern(layer_scale):
    # Pattern: double unsqueeze operations that expand scalar parameters to match spatial dimensions
    # This matches: layer_scale.unsqueeze(-1).unsqueeze(-1)
    # Original: [48] -> [48, 1] -> [48, 1, 1]
    tmp_8 = layer_scale.unsqueeze(-1)
    tmp_9 = tmp_8.unsqueeze(-1)
    return tmp_9

def replacement_args(layer_scale):
    return (layer_scale,)

@triton.jit
def expand_layer_scale_kernel(
    layer_scale_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial location
    pid = tl.program_id(0)
    total_elements = batch_size * channels * height * width
    grid_size = tl.cdiv(total_elements, BLOCK_SIZE)
    
    if pid >= grid_size:
        return
        
    # Calculate element index
    element_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = element_idx < total_elements
    
    # Load layer_scale - it has shape [batch * channels]
    layer_scale_idx = element_idx // (height * width)  # Map to corresponding (batch, channel)
    layer_scale_mask = layer_scale_idx < (batch_size * channels)
    
    layer_scale_val = tl.load(layer_scale_ptr + layer_scale_idx, mask=layer_scale_mask, other=0.0)
    
    # Since we want output to have shape [batch, channels, height, width]
    # and layer_scale_val has shape [batch, channels] after broadcasting,
    # we just need to broadcast it to the full spatial dimensions
    # In Triton, this natural broadcasting works automatically
    
    # Store the result - just broadcast the loaded values
    tl.store(output_ptr + element_idx, layer_scale_val, mask=mask)

@torch.fx.wrap
def optimized_layer_scale_expand(layer_scale, target_shape=None):
    """
    Optimized replacement for double unsqueeze operations.
    Directly broadcasts layer_scale to target spatial dimensions.
    """
    # Get target shape from input if not provided
    if target_shape is None:
        # For typical use case, expand to match spatial dimensions
        # Original pattern creates [48, 1, 1] which broadcasts to [batch, channels, height, width]
        expanded = layer_scale.unsqueeze(-1).unsqueeze(-1)
        target_shape = expanded.shape
    
    # Original pattern creates [channels, 1, 1] when layer_scale is [channels]
    if len(layer_scale.shape) == 1:
        expanded = layer_scale.unsqueeze(-1).unsqueeze(-1)
    else:
        expanded = layer_scale
    
    return expanded

@triton.jit
def direct_expand_kernel(
    layer_scale_ptr,
    output_ptr,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Optimized kernel that directly expands [channels] to [channels, height, width]
    pid = tl.program_id(0)
    total_elements = channels * height * width
    grid_size = tl.cdiv(total_elements, BLOCK_SIZE)
    
    if pid >= grid_size:
        return
        
    element_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = element_idx < total_elements
    
    # Calculate indices for output tensor [channels, height, width]
    channel = element_idx // (height * width)
    spatial = element_idx % (height * width)
    w = spatial % width
    h = spatial // width
    
    # Load layer_scale value for this channel
    layer_scale_val = tl.load(layer_scale_ptr + channel, mask=channel < channels, other=0.0)
    
    # Store the value (broadcasted across spatial dimensions)
    tl.store(output_ptr + element_idx, layer_scale_val, mask=mask)

@torch.fx.wrap  
def optimized_layer_scale_expand(layer_scale):
    """Memory-efficient expansion that minimizes overhead"""
    
    # The key insight is that for small tensors, we want to avoid function call overhead
    # but also avoid creating unnecessary intermediate tensors
    
    channels = layer_scale.shape[0]
    
    # For very small channel counts (like 48), the most efficient approach is to
    # create the expanded tensor directly in one step
    # This approach is simple and has minimal overhead
    
    # Create output with the final shape directly - this avoids intermediate tensor allocations
    expanded_shape = (channels, 1, 1)
    return layer_scale.reshape(channels, 1, 1)

def replacement_func():
    return optimized_layer_scale_expand