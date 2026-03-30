import torch
import triton
import triton.language as tl

def pattern(x, y):
    return x * y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def multiply_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * channels * height * width
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x * y
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_multiply(x, y):
    # Handle broadcasting - make sure x and y are compatible
    if x.shape != y.shape:
        # This is a simplified handling - real implementation would need more sophisticated broadcasting
        if x.numel() == 1:  # x is a scalar
            N = y.numel()
            BLOCK_SIZE = 1024
            num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
            
            out = torch.empty_like(y)
            
            multiply_kernel[(num_programs,)](
                x_ptr=x,
                y_ptr=y,
                out_ptr=out,
                batch_size=y.shape[0],
                channels=y.shape[1] if len(y.shape) > 1 else 1,
                height=y.shape[2] if len(y.shape) > 2 else 1,
                width=y.shape[3] if len(y.shape) > 3 else 1,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return out
        elif y.numel() == 1:  # y is a scalar
            N = x.numel()
            BLOCK_SIZE = 1024
            num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
            
            out = torch.empty_like(x)
            
            multiply_kernel[(num_programs,)](
                x_ptr=x,
                y_ptr=y,
                out_ptr=out,
                batch_size=x.shape[0],
                channels=x.shape[1] if len(x.shape) > 1 else 1,
                height=x.shape[2] if len(x.shape) > 2 else 1,
                width=x.shape[3] if len(x.shape) > 3 else 1,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return out
        else:
            # For non-scalar broadcasting, fall back to PyTorch
            return x * y
    else:
        # Same shapes - optimized path
        N = x.numel()
        BLOCK_SIZE = 1024
        num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        out = torch.empty_like(x)
        
        multiply_kernel[(num_programs,)](
            x_ptr=x,
            y_ptr=y,
            out_ptr=out,
            batch_size=x.shape[0],
            channels=x.shape[1] if len(x.shape) > 1 else 1,
            height=x.shape[2] if len(x.shape) > 2 else 1,
            width=x.shape[3] if len(x.shape) > 3 else 1,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return out

def replacement_func():
    return optimized_multiply