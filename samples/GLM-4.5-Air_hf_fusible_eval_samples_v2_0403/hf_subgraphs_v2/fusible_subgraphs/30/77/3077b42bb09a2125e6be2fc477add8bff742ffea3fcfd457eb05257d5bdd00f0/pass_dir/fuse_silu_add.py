import torch
import triton
import triton.language as tl

def pattern(x, y):
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_silu_add_kernel(
    x_ptr,  # in_0 pointer 
    y_ptr,  # in_1 pointer (for SiLU)
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Cast inputs to fp32 for better numerical stability in exponential operations
    x_fp32 = tl.cast(x, tl.float32)
    y_fp32 = tl.cast(y, tl.float32)
    
    # Compute fused SiLU + Add: silu(y) + x = y * sigmoid(y) + x
    # Use simple sigmoid computation for numerical stability
    sigmoid_y = 1.0 / (1.0 + tl.exp(-y_fp32))
    
    # Compute silu(y) and add x, all in fp32
    silu_y = y_fp32 * sigmoid_y
    out_fp32 = silu_y + x_fp32
    
    # Cast back to original precision
    out = tl.cast(out_fp32, y.dtype)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_silu_add(x, y):
    # Determine dtype from inputs
    dtype = x.dtype
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Get total number of elements
    n_elements = x.numel()
    
    # Use power-of-2 block sizes for Triton compatibility
    if dtype == torch.float32:
        BLOCK_SIZE = 1024  # Use 1024 for fp32
    else:
        BLOCK_SIZE = 2048  # Use 2048 for fp16/bf16
    
    # Calculate number of programs
    num_programs = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    fused_silu_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y, 
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_silu_add