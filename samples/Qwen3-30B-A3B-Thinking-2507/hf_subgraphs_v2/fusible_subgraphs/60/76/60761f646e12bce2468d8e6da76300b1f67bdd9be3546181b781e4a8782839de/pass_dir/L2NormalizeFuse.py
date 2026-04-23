import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0):
    # The cat operation is redundant (concatenating single tensor) but must match graph structure
    cat_out = torch.cat([in_0], 1)
    result = torch.nn.functional.normalize(cat_out, p=2, dim=1)
    return result

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Triton kernel
@triton.jit
def normalize_kernel(x_ptr, out_ptr, batch_size, BLOCK_SIZE: tl.constexpr):
    # Each block handles one row
    row_idx = tl.program_id(0)
    row_start = row_idx * BLOCK_SIZE
    
    # Load the entire row (768 elements)
    x = tl.load(x_ptr + row_start, shape=(BLOCK_SIZE,))
    
    # Compute sum of squares (with loop unrolling)
    sum_sq = 0.0
    for i in range(BLOCK_SIZE):
        x_i = x[i]
        sum_sq += x_i * x_i
    
    # Compute norm (handle zero vectors)
    norm = tl.sqrt(sum_sq)
    norm = tl.where(norm > 1e-7, norm, 1.0)
    
    # Normalize and store
    for i in range(BLOCK_SIZE):
        out = x[i] / norm
        tl.store(out_ptr + row_start + i, out)

# Kernel wrapper
@torch.fx.wrap
def l2_normalize(in_0):
    batch_size, dim = in_0.shape
    out = torch.empty_like(in_0)
    grid = (batch_size,)
    normalize_kernel[grid](in_0, out, batch_size, dim)
    return out

def replacement_func():
    return l2_normalize