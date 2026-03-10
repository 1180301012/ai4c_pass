import torch
import triton
import triton.language as tl


def pattern(x, y):
    tmp_0 = x / 8.0
    tmp_2 = tmp_0 + y
    return tmp_2


def replacement_args(x, y):
    return (x, y)


@triton.jit
def fused_div_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors with vectorization
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operation: (x / 8.0) + y
    out = (x * 0.125) + y  # Use multiplication instead of division for better performance
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


# Autotune configurations for different tensor sizes
@triton.heuristics({
    'BLOCK_SIZE': lambda args: 256 if args['n_elements'] <= 2048 else 1024
})
@triton.jit
def autotuned_fused_div_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operation: (x / 8.0) + y
    out = (x * 0.125) + y  # Use multiplication instead of division for better performance
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_fused_div_add(x, y):
    # Calculate total number of elements
    n_elements = x.numel()
    
    # Use heuristic-based block size selection for better performance
    if n_elements <= 2048:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch the autotuned kernel
    autotuned_fused_div_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return triton_fused_div_add