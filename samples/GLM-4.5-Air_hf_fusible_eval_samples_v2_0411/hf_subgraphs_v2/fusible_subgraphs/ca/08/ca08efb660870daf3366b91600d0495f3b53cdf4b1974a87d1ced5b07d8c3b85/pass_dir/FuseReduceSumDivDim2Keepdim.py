import torch
import triton
import triton.language as tl

def pattern(in_1):
    tmp_0 = in_1.sum(dim=2, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return (tmp_1,)

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def normalize_kernel(
    in_ptr,
    out_ptr,
    n_batch,
    n_channels,
    h, w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= n_batch * n_channels:
        return
    
    # Calculate thread offsets
    batch_idx = pid // n_channels
    channel_idx = pid % n_channels
    
    in_ptr_base = in_ptr + batch_idx * n_channels * h * w + channel_idx * w
    out_ptr_base = out_ptr + batch_idx * n_channels * h * w + channel_idx * w
    
    # Calculate sum along w dimension (dim=2)
    sum_val = 0.0
    
    # Sum the entire w dimension
    for i in range(w):
        offset = i
        val = tl.load(in_ptr_base + offset)
        sum_val += val
    
    # Normalize (handle potential division by zero)
    if sum_val == 0:
        sum_val = 1.0
    
    # Normalize across w dimension and store in place
    for pos in range(w):
        offset = pos
        val = tl.load(in_ptr_base + offset)
        out_val = val / sum_val
        tl.store(out_ptr_base + offset, out_val)

@torch.fx.wrap
def normalize_launcher(in_1):
    # Input shape: [1, 2, 8, 8]
    n_batch, n_channels, h, w = in_1.shape
    n_elements = n_batch * n_channels
    out_shape = (n_batch, n_channels, h, w)
    
    out = torch.empty(out_shape, dtype=in_1.dtype, device=in_1.device)
    
    BLOCK_SIZE = 1024
    if n_elements < 128:
        BLOCK_SIZE = 128
    
    grid = (n_elements,)
    
    normalize_kernel[grid](
        in_1,
        out,
        n_batch,
        n_channels,
        h, w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return normalize_launcher