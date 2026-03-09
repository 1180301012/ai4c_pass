import torch
import triton
import triton.language as tl

def pattern(x, p):
    # Pattern for dropout operation
    return torch.nn.functional.dropout(x, p, False, False)

def replacement_args(x, p):
    return (x, p)

@triton.jit
def dropout_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    if mask:
        # Load input data
        x_data = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Fast path for p == 0.0 (no-op)
        if p == 0.0:
            tl.store(out_ptr + offsets, x_data, mask=mask)
        else:
            # Generate random numbers using XORShift for better performance
            seed = 12345 + pid  # Deterministic seed per block
            state = seed
            
            for i in tl.arange(0, BLOCK_SIZE):
                if offsets[i] < n_elements:
                    # XORShift random number generator
                    state ^= state << 13
                    state ^= state >> 17
                    state ^= state << 5
                    rand_val = state / 2**32  # Normalize to [0,1)
                    
                    # Apply dropout scaling
                    scale = 1.0 / (1.0 - p)
                    if rand_val > p:
                        out_val = x_data[i] * scale
                    else:
                        out_val = 0.0
                    
                    tl.store(out_ptr + offsets[i], out_val, mask=mask)

@triton.jit
def optimized_dropout_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    # More efficient version with better random number generation
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    if tl.any(mask):
        x_data = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Fast path for p == 0.0
        if p == 0.0:
            tl.store(out_ptr + offsets, x_data, mask=mask)
            return
        
        # Generate random numbers using tl.rand for better performance
        rand_vals = tl.rand(offsets, mask=mask)
        
        # Scale factor
        scale = 1.0 / (1.0 - p)
        
        # Apply dropout
        mask_keep = rand_vals > p
        out_vals = tl.where(mask_keep, x_data * scale, 0.0)
        
        tl.store(out_ptr + offsets, out_vals, mask=mask)

@torch.fx.wrap
def optimized_dropout(x, p, training=True):
    # If not training or p == 0.0, return input directly (no-op)
    if not training or p == 0.0 or p == 1.0:
        return x
    
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_dropout_kernel[grid_size](
        x, output, n_elements, p, BLOCK_SIZE
    )
    
    return output

def optimized_dropout_wrapper(x, p):
    # Wrapper that matches the signature pattern(x, p)
    if p == 0.0 or p == 1.0:
        return x
    return optimized_dropout(x, p, training=True)

def replacement_func():
    return optimized_dropout_wrapper