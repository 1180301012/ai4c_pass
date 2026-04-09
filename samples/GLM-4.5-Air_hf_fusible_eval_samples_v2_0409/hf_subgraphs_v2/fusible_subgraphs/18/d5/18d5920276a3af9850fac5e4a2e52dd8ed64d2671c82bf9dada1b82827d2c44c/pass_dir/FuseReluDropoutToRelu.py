import torch
import triton
import triton.language as tl

def pattern(in_0):
    # Match the sequence: ReLU -> Dropout(p=0.0) 
    tmp_0 = torch.nn.functional.relu(in_0, inplace=False)
    tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
    return tmp_1

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def optimized_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized ReLU kernel that eliminates the no-op dropout"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(x, 0)
    out = tl.maximum(x, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_dropout(in_0):
    """Fused ReLU + Dropout (p=0.0) -> effectively just ReLU"""
    n_elements = in_0.numel()
    block_size = 1024  # Optimal block size for most GPUs
    num_programs = (n_elements + block_size - 1) // block_size
    
    # Create output with same shape and dtype as input
    out = torch.empty_like(in_0)
    
    # Launch Triton kernel
    optimized_relu_kernel[(num_programs,)](
        x_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=block_size
    )
    
    return out

def replacement_func():
    return fused_relu_dropout