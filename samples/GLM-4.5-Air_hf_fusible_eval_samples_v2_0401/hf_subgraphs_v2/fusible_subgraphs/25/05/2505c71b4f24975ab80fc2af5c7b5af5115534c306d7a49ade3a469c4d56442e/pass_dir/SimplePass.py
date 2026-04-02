import torch
import triton
import triton.language as tl

def pattern(x, y):
    return x * y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def atomic_mul_kernel(
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
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=1.0)
    out = x * y
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_mul(x, y):
    # Ensure both inputs are flattened to 1D for elemental operations
    # This handles broadcasting by flattening the tensors    
    if x.numel() == 1:
        # x is scalar apply to every element of y
        n_elements = y.numel()
        out = torch.empty_like(y)
        if n_elements > 0:
            BLOCK_SIZE = 1024
            num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
            atomic_mul_kernel[(num_programs,)](
                x_ptr=x.reshape(-1),
                y_ptr=y.reshape(-1),
                out_ptr=out.reshape(-1), 
                n_elements=n_elements,
                BLOCK_SIZE=BLOCK_SIZE
            )
        return out
    elif y.numel() == 1:
        # y is scalar apply to every element of x  
        n_elements = x.numel()
        out = torch.empty_like(x)
        if n_elements > 0:
            BLOCK_SIZE = 1024
            num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
            atomic_mul_kernel[(num_programs,)](
                x_ptr=x.reshape(-1),
                y_ptr=y.reshape(-1),
                out_ptr=out.reshape(-1),
                n_elements=n_elements,
                BLOCK_SIZE=BLOCK_SIZE
            )
        return out
    else:
        # Both are tensors, need broadcasting - flatten and let Triton handle it
        # Use larger block size for tensor operations
        n_elements = max(x.numel(), y.numel())
        # Determine output shape using broadcasting
        out_shape = []
        for xa, ya in zip(x.shape[::-1], y.shape[::-1]):
            out_shape.append(max(xa, ya))
        out_shape = out_shape[::-1]
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        
        if n_elements > 0:
            BLOCK_SIZE = 1024
            num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
            atomic_mul_kernel[(num_programs,)](
                x_ptr=x.reshape(-1),
                y_ptr=y.reshape(-1),
                out_ptr=out.reshape(-1),
                n_elements=n_elements,
                BLOCK_SIZE=BLOCK_SIZE
            )
        return out

def replacement_func():
    return triton_mul