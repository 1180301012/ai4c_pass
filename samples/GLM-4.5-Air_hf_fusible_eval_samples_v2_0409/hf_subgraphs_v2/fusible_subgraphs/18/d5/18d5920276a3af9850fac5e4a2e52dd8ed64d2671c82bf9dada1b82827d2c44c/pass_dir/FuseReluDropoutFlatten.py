import torch
import triton
import triton.language as tl

def pattern(in_0):
    # Match the full sequence: ReLU -> Dropout(p=0.0) -> Flatten(1, -1)
    tmp_0 = torch.nn.functional.relu(in_0, inplace=False)
    tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
    tmp_2 = tmp_1.flatten(1, -1)
    return tmp_2

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def optimized_fused_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel that fuses ReLU (with dropout elimination) and flatten.
    Input shape: [batch_size, n_channels, 1, 1]
    Output shape: [batch_size, n_channels]
    """
    # Each program handles a column in the flattened 2D output [batch_size, n_channels]
    program_id = tl.program_id(0)
    channel_idx = program_id % n_channels
    batch_idx = program_id // n_channels
    
    # Calculate the flattening offset: since we have [batch, channels, 1, 1]
    # The flatten(1, -1) operation just removes the last two dimensions
    # We can access elements directly as if they were 2D [batch, channels]
    x_offset = batch_idx * n_channels + channel_idx
    out_offset = batch_idx * n_channels + channel_idx
    
    # Load input (accessing the 4D tensor as if it were flattened 2D)
    x = tl.load(x_ptr + x_offset, other=0.0)
    
    # Apply ReLU (dropout with p=0.0 is identity, so we skip it)
    out = tl.maximum(x, 0.0)
    
    # Store result (directly in 2D format without reshape)
    tl.store(out_ptr + out_offset, out)

@torch.fx.wrap
def fused_relu_dropout_flatten(in_0):
    """Fused kernel for ReLU + Dropout(p=0.0) + Flatten(1, -1)"""
    # Input shape: [batch_size, n_channels, 1, 1]
    # After flatten: [batch_size, n_channels]
    batch_size, n_channels, _, _ = in_0.shape
    n_elements = batch_size * n_channels
    
    # Use larger block size for better GPU occupancy
    block_size = 1024
    num_programs = (n_elements + block_size - 1) // block_size
    
    # Create output with flattened shape [batch_size, n_channels]
    out_shape = (batch_size, n_channels)
    out = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Launch Triton kernel
    optimized_fused_kernel[(num_programs,)](
        x_ptr=in_0,
        out_ptr=out,
        batch_size=batch_size,
        n_channels=n_channels,
        BLOCK_SIZE=block_size
    )
    
    return out

def replacement_func():
    return fused_relu_dropout_flatten