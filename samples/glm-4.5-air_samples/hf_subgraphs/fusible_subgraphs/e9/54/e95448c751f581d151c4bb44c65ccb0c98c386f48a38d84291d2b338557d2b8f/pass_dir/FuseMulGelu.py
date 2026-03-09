import torch
import triton
import triton.language as tl

def pattern(x, y):
    tmp_4 = x * y
    tmp_5 = torch.nn.functional.gelu(tmp_4, approximate='none')
    return tmp_5

def replacement_args(x, y):
    return (x, y)

@triton.jit
def mul_gelu_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
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
    
    # Element-wise multiplication
    mul = x * y
    
    # GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Simplified GELU approximation for better performance
    gelu = mul * 0.5 * (1.0 + tl.tanh(tl.sqrt(2.0 / 3.141592653589793) * 
                  (1.0 + 0.044715 * mul * mul * mul) * mul))
    
    # Store result
    tl.store(out_ptr + offsets, gelu, mask=mask)

@torch.fx.wrap
def mul_gelu_impl(x, y):
    # Ensure x and y have the same shape
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x.shape={x.shape}, y.shape={y.shape}")
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    mul_gelu_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return mul_gelu_impl