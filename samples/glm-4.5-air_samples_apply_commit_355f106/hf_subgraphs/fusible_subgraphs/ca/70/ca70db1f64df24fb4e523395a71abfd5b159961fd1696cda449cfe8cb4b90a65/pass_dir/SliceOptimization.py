import torch
import triton
import triton.language as tl

def pattern(x):
    # pattern matches: slicing operations from linear output
    tmp_slice1 = x[slice(None, None, None), slice(None, 256, None)]
    tmp_slice2 = x[slice(None, None, None), slice(-256, None, None)]
    return tmp_slice1, tmp_slice2

def replacement_args(x):
    return (x,)

@triton.jit
def slice_kernel(
    x_ptr,
    out1_ptr,
    out2_ptr,
    n_batch,
    in_features,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element
    batch_idx = tl.program_id(0)
    x_offset = batch_idx * in_features
    
    # Load input
    x = tl.load(x_ptr + x_offset, mask=x_offset + tl.arange(0, in_features) < n_batch * in_features)
    
    # Extract slices
    slice1 = x[:256]
    slice2 = x[-256:]
    
    # Store slices
    tl.store(out1_ptr + batch_idx * 256, slice1)
    tl.store(out2_ptr + batch_idx * 256, slice2)

@torch.fx.wrap
def triton_slice(x):
    n_batch = x.shape[0]
    in_features = x.shape[1]
    BLOCK_SIZE = 256
    
    # Create output tensors
    out1 = torch.empty((n_batch, 256), dtype=x.dtype, device='cuda')
    out2 = torch.empty((n_batch, 256), dtype=x.dtype, device='cuda')
    
    # Launch kernel
    slice_kernel[(n_batch,)](
        x_ptr=x,
        out1_ptr=out1,
        out2_ptr=out2,
        n_batch=n_batch * in_features,
        in_features=in_features,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out1, out2

def replacement_func():
    return triton_slice