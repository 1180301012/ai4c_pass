import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Match the exact computation pattern from the model
    # Note: We must NOT include the None statements in pattern matching
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = in_1.view(in_1.shape[0], 512, -1)
    return (tmp_4, tmp_5)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_max_softmax_kernel(
    energy_ptr,
    softmax_out_ptr,
    batch_size,
    channel_size,
    h_dim,
    w_dim,
    BLOCK_W: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Compute max along the last dimension (w_dim)
    max_val = -float('inf')
    # Calculate starting position for this (batch, channel)
    base_offset = batch_idx * channel_size * h_dim * w_dim + channel_idx * h_dim * w_dim
    
    # Find max along width dimension
    for w in range(w_dim):
        offset = base_offset + w
        val = tl.load(energy_ptr + offset, other=-float('inf'))
        max_val = tl.maximum(max_val, val)
    
    # Compute softmax: exp(x - max) / sum(exp(x - max))
    sum_exp = 0.0
    for w in range(w_dim):
        offset = base_offset + w
        val = tl.load(energy_ptr + offset, other=-float('inf'))
        exp_val = tl.exp(val - max_val)
        sum_exp += exp_val
    
    # Store softmax results
    for w in range(w_dim):
        offset = base_offset + w
        val = tl.load(energy_ptr + offset, other=-float('inf'))
        exp_val = tl.exp(val - max_val)
        softmax_val = exp_val / sum_exp
        tl.store(softmax_out_ptr + offset, softmax_val)

@torch.fx.wrap
def optimized_forward(in_0, in_1):
    # Get input shapes
    batch_size, channel_size, h_dim, w_dim = in_0.shape[0], in_0.shape[1], in_1.shape[2], in_1.shape[3]
    
    # Create output tensor for softmax
    softmax_out = torch.empty_like(in_0)
    
    # Launch fused kernel for max + softmax
    grid = (batch_size, channel_size)
    BLOCK_W = 128  # Block size for width dimension
    
    fused_max_softmax_kernel[grid](
        in_0,
        softmax_out,
        batch_size,
        channel_size,
        h_dim,
        w_dim,
        BLOCK_W=BLOCK_W,
    )
    
    # Apply view operation to in_1 
    view_out = in_1.view(in_1.shape[0], 512, -1)
    
    return softmax_out, view_out

def replacement_func():
    return optimized_forward