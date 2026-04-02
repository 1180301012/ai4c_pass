import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Pattern matching multiplication followed by dropout"""
    tmp_1 = in_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_mul_dropout_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    """Fused multiplication + dropout kernel with data type optimization"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to stay within bounds
    
    # Load input tensors with type inference
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Element-wise multiplication (dropout with p=0.0 is optimized out)
    out = x * y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_mul_dropout_wrapper(in_0, in_1):
    """Wrapper function to launch the fused multiplication + dropout kernel with optimized configuration"""
    N = in_0.numel()
    
    # For our tensor size (263,168 elements), use optimal configuration
    if N < 10000:
        BLOCK_SIZE = 256
        num_warps = 1
    elif N < 100000:
        BLOCK_SIZE = 1024
        num_warps = 2
    else:
        BLOCK_SIZE = 2048  # Larger block size for our 263K+ elements
        num_warps = 4      # More warps for better GPU utilization
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_0)
    
    fused_mul_dropout_kernel[(num_programs,)](
        x_ptr=in_0,
        y_ptr=in_1,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return out

def replacement_func():
    """Returns the fused kernel wrapper function"""
    return fused_mul_dropout_wrapper