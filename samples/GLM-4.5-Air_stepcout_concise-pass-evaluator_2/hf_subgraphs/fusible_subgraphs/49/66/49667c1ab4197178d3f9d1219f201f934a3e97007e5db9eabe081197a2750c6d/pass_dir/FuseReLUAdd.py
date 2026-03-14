import torch
import triton
import triton.language as tl

def pattern(in_1, in_0):
    """Pattern matching: ReLU followed by element-wise addition"""
    tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
    tmp_1 = tmp_0 + in_0
    return tmp_0, tmp_1

def replacement_args(in_1, in_0):
    """Extract arguments for the replacement function"""
    return in_1, in_0

@triton.jit
def fused_relu_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    relu_only: tl.constexpr = False,
):
    """Fused ReLU + Add kernel: max(0, x) + y or just max(0, x)"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0) if not relu_only else tl.zeros(x.shape, dtype=x.dtype)
    
    # Compute: max(0, x) + y (or just max(0, x) if relu_only)
    relu_x = tl.maximum(x, 0.0)
    out = relu_x + y if not relu_only else relu_x
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_add(x, y):
    """Wrapper function for fused ReLU + Add operation"""
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    relu_result = torch.empty_like(x)
    
    # Compute both results in separate kernel launches
    fused_relu_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
        relu_only=False,
    )
    
    fused_relu_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=relu_result,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
        relu_only=True,
    )
    
    return relu_result, out

def replacement_func():
    """Return the fused function"""
    return fused_relu_add