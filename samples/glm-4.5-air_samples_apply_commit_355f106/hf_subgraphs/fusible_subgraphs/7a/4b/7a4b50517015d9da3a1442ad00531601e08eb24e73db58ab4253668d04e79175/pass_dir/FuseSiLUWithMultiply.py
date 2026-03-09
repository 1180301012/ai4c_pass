import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Match SiLU followed by element-wise multiplication
    tmp_0 = torch.nn.functional.silu(in_0, inplace=False)
    tmp_1 = tmp_0 * in_1
    return tmp_1

def replacement_args(in_0, in_1):
    # Need both input tensors for the fused operation
    return (in_0, in_1)

@triton.jit
def fused_silu_multiply_kernel(
    x_ptr,           # in_0 pointer
    y_ptr,           # in_1 pointer  
    out_ptr,         # output pointer
    n_elements,      # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused operation: x * sigmoid(x) * y
    # SiLU(x) = x * sigmoid(x), so we do: x * sigmoid(x) * y
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    out = x * sigmoid_x * y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_silu_multiply(in_0, in_1):
    # Set up grid and launch kernel
    n_elements = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor
    out = torch.empty_like(in_0)
    
    # Launch kernel
    fused_silu_multiply_kernel[(num_programs,)](
        in_0,
        in_1,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_silu_multiply