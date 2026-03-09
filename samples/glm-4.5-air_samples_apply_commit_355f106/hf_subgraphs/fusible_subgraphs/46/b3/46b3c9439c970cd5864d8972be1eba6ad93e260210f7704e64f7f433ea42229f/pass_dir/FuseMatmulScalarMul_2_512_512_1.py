import torch
import triton
import triton.language as tl

def pattern(x, y, other):
    # Match: matmul + scalar multiplication pattern
    # x: [2, 512], y: [512, 1], other: scalar
    tmp_0 = torch.matmul(x, y)
    tmp_1 = tmp_0 * other
    return tmp_1

def replacement_args(matmul_input, matmul_other, scalar):
    return (matmul_input, matmul_other, scalar)

@triton.jit
def optimized_fused_kernel(
    x_ptr, y_ptr, scalar_ptr,
    out_ptr,
    M: tl.constexpr, K: tl.constexpr
):
    # Each program handles one row of the output
    row_idx = tl.program_id(0)
    
    if row_idx >= M:
        return
    
    # Load scalar value once
    scalar = tl.load(scalar_ptr)
    
    # Optimized: Use a single loop with vectorization when possible
    # For this small size (K=512), we can optimize the memory access
    acc = 0.0
    
    # Optimized: For this small size (K=512), vectorized operations work best
    # We'll load the entire row and compute in one go
    idx = tl.arange(0, K)
    
    # Load x row and y vector with proper masking
    x_row = tl.load(x_ptr + row_idx * K + idx, mask=idx < K, other=0.0)
    y_vec = tl.load(y_ptr + idx, mask=idx < K, other=0.0)
    
    # Compute dot product using vectorized operations
    acc = tl.sum(x_row * y_vec)
    
    # Apply scalar and store result
    result = acc * scalar
    tl.store(out_ptr + row_idx, result)

@torch.fx.wrap
def fused_matmul_scalar_func(x, y, scalar):
    # Specialized for our known shapes: x[2, 512], y[512, 1], output[2, 1]
    M, K = x.shape
    
    # Create output tensor with shape [M, 1]
    out = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    out_flat = out.view(-1)  # Flatten to [M] for simpler storage
    
    # Launch kernel with optimized block processing
    grid_size = (M,)
    
    optimized_fused_kernel[grid_size](
        x_ptr=x,
        y_ptr=y,
        scalar_ptr=scalar,
        out_ptr=out_flat,
        M=M, K=K
    )
    
    return out

def replacement_func():
    return fused_matmul_scalar_func