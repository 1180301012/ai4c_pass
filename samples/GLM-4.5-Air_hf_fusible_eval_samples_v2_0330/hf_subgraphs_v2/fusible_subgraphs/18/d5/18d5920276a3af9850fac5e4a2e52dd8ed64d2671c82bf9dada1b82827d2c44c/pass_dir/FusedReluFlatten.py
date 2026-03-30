import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace = False)
    tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
    tmp_2 = tmp_1.flatten(1, -1)
    return tmp_2

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_1d_flatten_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    BLOCK_SIZE: tl.constexpr,
):
    # Program handles a portion of the batch * channels flattened tensor
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * channels
    
    # Load data (assuming 4D tensor [batch, channels, 1, 1])
    batch_idx = offsets // channels
    channel_idx = offsets % channels
    
    # Load from 4D tensor layout
    x = tl.load(x_ptr + batch_idx * channels + channel_idx,
                mask=batch_idx < batch_size,
                other=0.0)
    
    # Apply ReLU
    out = tl.maximum(x, 0.0)
    
    # Store (already flattened)
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_flatten_v2(x):
    if x.dim() != 4 or x.shape[2] != 1 or x.shape[3] != 1:
        # Fallback to original for non-[batch, channels, 1, 1] shapes
        tmp_0 = torch.nn.functional.relu(x, inplace=False)
        tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
        tmp_2 = tmp_1.flatten(1, -1)
        return tmp_2
    
    batch_size, channels, _, _ = x.shape
    out = torch.empty([batch_size, channels], dtype=x.dtype, device=x.device)
    
    N = batch_size * channels
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_1d_flatten_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_relu_flatten_v2