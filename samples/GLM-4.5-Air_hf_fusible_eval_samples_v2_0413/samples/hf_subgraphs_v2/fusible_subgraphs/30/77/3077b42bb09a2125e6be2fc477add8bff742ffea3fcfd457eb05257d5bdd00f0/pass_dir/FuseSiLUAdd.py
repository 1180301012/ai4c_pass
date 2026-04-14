import torch
import triton
import triton.language as tl

def pattern(a, b):
    """
    Simple pattern to understand matching - just Addition
    """
    result = a + b
    return result

def replacement_args(a, b):
    return (a, b)

# Triton kernel for fused SiLU + Add
@triton.jit
def silu_add_kernel(
    x_ptr,
    y_ptr, 
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused SiLU + Add
    # SiLU(x) = x * sigmoid(x) = x * (1/(1+exp(-x)))
    # Cast to fp32 for exponential operations
    x_fp32 = x.to(tl.float32)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x_fp32))
    
    # Cast back to original dtype for multiplication
    if x.dtype == tl.float16 or x.dtype == tl.bfloat16:
        sigmoid_x = sigmoid_x.to(x.dtype)
    
    silu_x = x * sigmoid_x
    out = silu_x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_fused_silu_add(x, y):
    """
    Triton-optimized fused SiLU + Add operation
    Computes: (x * sigmoid(x)) + y
    """
    # Handle different tensor shapes
    if x.dim() == 4:  # Common case for 2D convolutions
        N, C, H, W = x.shape
        n_elements = N * C * H * W
        out = torch.empty_like(x)
        BLOCK_SIZE = 1024
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        silu_add_kernel[(num_programs,)](
            x_ptr=x,
            y_ptr=y,
            out_ptr=out,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Fallback for other shapes
        n_elements = x.numel()
        out = torch.empty_like(x)
        BLOCK_SIZE = 1024
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        silu_add_kernel[(num_programs,)](
            x_ptr=x,
            y_ptr=y,
            out_ptr=out,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

def replacement_func():
    return triton_fused_silu_add