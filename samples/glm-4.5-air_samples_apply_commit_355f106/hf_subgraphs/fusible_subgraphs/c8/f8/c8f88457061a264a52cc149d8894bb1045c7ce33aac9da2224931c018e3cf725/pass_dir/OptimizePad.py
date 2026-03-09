import torch
import triton
import triton.language as tl

def pattern(x):
    pad_out = torch.nn.functional.pad(x, (0, 1, 0, 1), 'constant', None)
    return pad_out

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_pad_kernel(
    x_ptr,
    out_ptr,
    n_elements_batch,
    n_elements_channel,
    input_height,
    input_width,
    output_height,
    output_width,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Each program handles a spatial tile
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    h_idx = tl.program_id(2) * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_idx = tl.program_id(3) * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for bounds checking
    h_mask = h_idx < output_height
    w_mask = w_idx < output_width
    
    # Calculate input indices (accounting for padding)
    # We pad (0, 1) on last dimension and (0, 1) on second-to-last dimension
    input_h = h_idx
    input_w = w_idx
    
    # Compute flat indices
    batch_offset = batch_idx * n_elements_channel
    channel_offset = channel_idx * input_height * input_width
    
    in_indices = (
        batch_offset +
        channel_offset +
        input_h * input_width +
        input_w
    )
    
    out_indices = (
        batch_idx * n_elements_channel +
        channel_idx * output_height * output_width +
        h_idx * output_width +
        w_idx
    )
    
    # Read from input (use other=0.0 for padded regions)
    x = tl.load(x_ptr + in_indices, mask=h_mask & w_mask, other=0.0)
    
    # Write to output
    tl.store(out_ptr + out_indices, x, mask=h_mask & w_mask)

@triton.autotune(
    configs=[
        triton.Config(num_warps=4, num_stages=3, kwargs={}),
        triton.Config(num_warps=8, num_stages=3, kwargs={}),
        triton.Config(num_warps=8, num_stages=4, kwargs={}),
        triton.Config(num_warps=16, num_stages=4, kwargs={}),
    ],
    key=['input_height', 'input_width', 'output_height', 'output_width'],
)
@triton.jit
def optimized_pad_autotuned_kernel(
    x_ptr,
    out_ptr,
    n_elements_batch,
    n_elements_channel,
    input_height,
    input_width,
    output_height,
    output_width,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Each program handles a spatial tile
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    h_idx = tl.program_id(2) * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_idx = tl.program_id(3) * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for bounds checking
    h_mask = h_idx < output_height
    w_mask = w_idx < output_width
    
    # Calculate input indices (accounting for padding)
    # We pad (0, 1) on last dimension and (0, 1) on second-to-last dimension
    input_h = h_idx
    input_w = w_idx
    
    # Compute flat indices
    batch_offset = batch_idx * n_elements_channel
    channel_offset = channel_idx * input_height * input_width
    
    in_indices = (
        batch_offset +
        channel_offset +
        input_h * input_width +
        input_w
    )
    
    out_indices = (
        batch_idx * n_elements_channel +
        channel_idx * output_height * output_width +
        h_idx * output_width +
        w_idx
    )
    
    # Read from input (use other=0.0 for padded regions)
    x = tl.load(x_ptr + in_indices, mask=h_mask & w_mask, other=0.0)
    
    # Write to output
    tl.store(out_ptr + out_indices, x, mask=h_mask & w_mask)

@torch.fx.wrap
def optimized_pad(x):
    B, C, H, W = x.shape
    out_shape = (B, C, H + 1, W + 1)  # Add (0,1) padding on height and width
    
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # For input tensors with small dimensions, use optimized autotuned kernel
    if H * W <= 8192:  # Threshold for small tensors
        optimized_pad_autotuned_kernel[
            (B, C, (H + 63) // 64, (W + 63) // 64),
            (1, 1, 64, 64),
        ](
            x_ptr=x,
            out_ptr=out,
            n_elements_batch=B * C * (H + 1) * (W + 1),
            n_elements_channel=(H + 1) * (W + 1),
            input_height=H,
            input_width=W,
            output_height=H + 1,
            output_width=W + 1,
            BLOCK_SIZE_H=64,
            BLOCK_SIZE_W=64,
        )
    else:
        # For larger tensors, use grid-strided approach
        n_elements_batch = B * C * (H + 1) * (W + 1)
        n_elements_channel = (H + 1) * (W + 1)
        
        optimized_pad_kernel[
            (B, C, (H + 1 + 63) // 64, (W + 1 + 63) // 64),
            (1, 1, 64, 64),
        ](
            x_ptr=x,
            out_ptr=out,
            n_elements_batch=n_elements_batch,
            n_elements_channel=n_elements_channel,
            input_height=H,
            input_width=W,
            output_height=H + 1,
            output_width=W + 1,
            BLOCK_SIZE_H=64,
            BLOCK_SIZE_W=64,
        )
    
    return out

def replacement_func():
    return optimized_pad