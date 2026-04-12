import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_2 = in_0.view(1, 2, 1, 8, 8)
    tmp_3 = tmp_2.expand(1, 2, 64, 8, 8)
    return (tmp_3,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def broadcast_view_kernel(
    in_ptr,
    out_ptr,
    n_batch,
    n_channels,
    h, w,
    out_h,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= n_batch * n_channels * out_h:
        return
    
    # Calculate thread offsets
    batch_idx = pid // (n_channels * out_h)
    remainder = pid % (n_channels * out_h)
    channel_idx = remainder // out_h
    h_idx = remainder % out_h
    
    # Calculate base pointers
    in_batch_offset = batch_idx * n_channels * h * w
    out_batch_offset = batch_idx * n_channels * out_h * w
    
    in_channel_offset = channel_idx * h * w
    out_channel_offset = channel_idx * out_h * w
    
    in_ptr_base = in_ptr + in_batch_offset + in_channel_offset + h_idx * w
    out_ptr_base = out_ptr + out_batch_offset + out_channel_offset + h_idx * w
    
    # Load and store for the specific H position
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < w
    
    # Load input data (all H positions are the same due to broadcasting)
    inp = tl.load(in_ptr_base + offsets, mask=mask, other=0.0)
    
    # Store output for the specific H position (broadcasting repeats same data)
    tl.store(out_ptr_base + offsets, inp, mask=mask)

@torch.fx.wrap
def broadcast_view_launcher(in_0):
    # Input shape: [1, 2, 8, 8]
    # Output shape: [1, 2, 64, 8, 8]
    n_batch, n_channels, h, w = in_0.shape
    out_h = 64
    
    # Reshape input to [1, 2, 64, 8, 8] by broadcasting
    out_shape = (n_batch, n_channels, out_h, h, w)
    out = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    
    BLOCK_SIZE = 1024
    if w < BLOCK_SIZE:
        BLOCK_SIZE = 512
        if w < 512:
            BLOCK_SIZE = 256
            if w < 256:
                BLOCK_SIZE = 128
    
    total_elements = n_batch * n_channels * out_h
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    broadcast_view_kernel[grid](
        in_0,
        out,
        n_batch,
        n_channels,
        h, w,
        out_h,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return broadcast_view_launcher