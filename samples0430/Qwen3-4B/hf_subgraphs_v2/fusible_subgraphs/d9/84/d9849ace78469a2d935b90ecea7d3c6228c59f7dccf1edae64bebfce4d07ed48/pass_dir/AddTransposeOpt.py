import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    added = in_1 + in_0
    return added.transpose(1, 2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_features,
    n_sequence,
    BLOCK_SIZE: tl.constexpr,
):
    f_idx = tl.program_id(0)
    s_idx = tl.program_id(1)
    
    if f_idx < n_features and s_idx < n_sequence:
        in_0_val = tl.load(in_0_ptr + s_idx, other=0.0)
        in_1_val = tl.load(in_1_ptr + s_idx * n_features + f_idx, other=0.0)
        out_val = in_0_val + in_1_val
        tl.store(out_ptr + f_idx * n_sequence + s_idx, out_val)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    in_0_shape = in_0.shape
    in_1_shape = in_1.shape
    n_features = in_1_shape[2]
    n_sequence = in_1_shape[1]
    n_elements = n_features * n_sequence
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty((1, n_features, n_sequence), dtype=in_0.dtype, device=in_0.device)
    
    optimized_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_features=n_features,
        n_sequence=n_sequence,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty(n_elements, dtype=in_0.dtype, device=in_0.device)
    
    optimized_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return kernel_wrapper