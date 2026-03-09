import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    # pattern matches: linear + slice + view operations
    tmp = torch.nn.functional.linear(x, weight, bias)
    tmp_slice1 = tmp[slice(None, None, None), slice(None, 256, None)]
    tmp_view1 = tmp_slice1.view(-1, 256)
    tmp_slice2 = tmp[slice(None, None, None), slice(-256, None, None)]
    tmp_view2 = tmp_slice2.view(-1, 256)
    return tmp_view1, tmp_view2

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def linear_slice_view_kernel(
    x_ptr,
    weight_ptr, 
    bias_ptr,
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
    
    # First half [0:256]
    tl.store(out1_ptr + batch_idx * 256, x[:256])
    
    # Second half [-256:] 
    tl.store(out2_ptr + batch_idx * 256, x[-256:])

@torch.fx.wrap
def triton_linear_slice_view(x, weight, bias):
    n_batch = x.shape[0]
    in_features = x.shape[1]
    BLOCK_SIZE = 256
    
    # Create output tensors
    out1 = torch.empty((n_batch, 256), dtype=x.dtype, device='cuda')
    out2 = torch.empty((n_batch, 256), dtype=x.dtype, device='cuda')
    
    # Launch kernel
    linear_slice_view_kernel[(n_batch,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out1_ptr=out1,
        out2_ptr=out2,
        n_batch=n_batch * in_features,
        in_features=in_features,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out1, out2

def replacement_func():
    return triton_linear_slice_view