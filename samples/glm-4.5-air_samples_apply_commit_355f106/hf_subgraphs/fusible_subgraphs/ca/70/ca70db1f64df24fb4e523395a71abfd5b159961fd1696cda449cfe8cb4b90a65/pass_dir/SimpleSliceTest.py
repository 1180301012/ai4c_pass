import torch
import triton
import triton.language as tl

def pattern(x):
    # Simple pattern: just a single slice operation
    result = x[slice(None, None, None), slice(None, 256, None)]
    return result

def replacement_args(x):
    return (x,)

@triton.jit
def simple_slice_kernel(
    x_ptr,
    out_ptr,
    n_batch,
    in_features: tl.constexpr,
    SLICE_SIZE: tl.constexpr,
):
    # Each program handles one batch element
    batch_idx = tl.program_id(0)
    x_offset = batch_idx * in_features
    
    # Extract first SLICE_SIZE elements 
    for i in tl.range(0, SLICE_SIZE):
        elem = tl.load(x_ptr + x_offset + i, mask=(x_offset + i < n_batch * in_features))
        tl.store(out_ptr + batch_idx * SLICE_SIZE + i, elem)

@torch.fx.wrap
def triton_simple_slice(x):
    n_batch = x.shape[0]
    in_features = x.shape[1]
    slice_size = 256
    
    out = torch.empty((n_batch, slice_size), dtype=x.dtype, device='cuda')
    
    simple_slice_kernel[(n_batch,)](
        x_ptr=x,
        out_ptr=out,
        n_batch=n_batch,
        in_features=in_features,
        SLICE_SIZE=slice_size
    )
    
    return out

def replacement_func():
    return triton_simple_slice