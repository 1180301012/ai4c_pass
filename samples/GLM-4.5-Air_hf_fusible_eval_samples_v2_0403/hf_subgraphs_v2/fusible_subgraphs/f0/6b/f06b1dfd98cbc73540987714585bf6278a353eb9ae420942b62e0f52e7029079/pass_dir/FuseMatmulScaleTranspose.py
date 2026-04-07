import torch
import triton
import triton.language as tl

def pattern(x, y, z):
    """Pattern matching: matmul + scale"""
    matmul = torch.matmul(z, y)
    result = matmul * x
    return result, result.t()

def replacement_args(x, y, z):
    """Extract arguments for the replacement"""
    return (x, y, z)

@triton.jit
def fused_matmul_scale_kernel(
    in_2_ptr,      # [2, 1024] - input matrix
    in_1_ptr,      # [1024, 1] - vector to multiply
    in_0,          # scalar - scale factor
    out_1_ptr,     # [2, 1] - first output
    out_2_ptr,     # [1, 2] - second output (transposed)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Handle first row of output [2, 1]
    if pid < 2:
        row_offset = pid * 1024
        sum_val = tl.zeros([1], dtype=tl.float32)
        
        # Vectorized reduction along the 1024 dimension
        for i in range(0, 1024, BLOCK_SIZE):
            block_offset = row_offset + i
            mask = (i + tl.arange(0, BLOCK_SIZE)) < 1024
            
            # Load block from in_2 [2, 1024]
            in_2_block = tl.load(in_2_ptr + block_offset, mask=mask, other=0.0)
            # Load corresponding element from in_1 [1024, 1]
            in_1_block = tl.load(in_1_ptr + i, mask=mask, other=0.0)
            
            # Multiply and accumulate
            sum_val += in_2_block * in_1_block
        
        # Apply scalar multiplication and store
        result = sum_val * in_0
        tl.store(out_1_ptr + pid, result)
        
        # Store transposed result directly to out_2 [1, 2]
        tl.store(out_2_ptr + pid, result)

@torch.fx.wrap
def fused_matmul_scale(in_0, in_1, in_2):
    """Fused kernel: matmul + scale + transpose elimination"""
    # Output shapes
    out_1_shape = [2, 1]    # Original result [2, 1]
    out_2_shape = [1, 2]    # Transposed result [1, 2]
    
    # Create output tensors
    out_1 = torch.empty(out_1_shape, device=in_2.device, dtype=in_2.dtype)
    out_2 = torch.empty(out_2_shape, device=in_2.device, dtype=in_2.dtype)
    
    # Set up Triton kernel launch
    BLOCK_SIZE = 128  # Optimized for 1024 size vector
    num_rows = 2      # Number of rows in output [2, 1]
    grid = (num_rows,)
    
    # Convert scalar to appropriate type
    if in_0.dtype == torch.float16:
        scale = float(in_0)
    elif in_0.dtype == torch.bfloat16:
        scale = float(in_0)
    else:
        scale = float(in_0)
    
    # Launch kernel
    fused_matmul_scale_kernel[grid](
        in_2_ptr=in_2,
        in_1_ptr=in_1, 
        in_0=scale,
        out_1_ptr=out_1,
        out_2_ptr=out_2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_1, out_2

def replacement_func():
    """Return the optimized kernel wrapper function"""
    return fused_matmul_scale