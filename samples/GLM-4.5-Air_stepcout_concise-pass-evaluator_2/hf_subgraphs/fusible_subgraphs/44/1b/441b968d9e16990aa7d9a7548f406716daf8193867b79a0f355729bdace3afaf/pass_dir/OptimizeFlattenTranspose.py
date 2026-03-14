import torch
import triton
import triton.language as tl

def pattern(x):
    """Match flatten followed by transpose"""
    tmp = x.flatten(2)
    result = tmp.transpose(1, 2)
    return result

def replacement_args(x):
    """Extract arguments for optimized kernel"""
    return (x,)

@triton.jit
def optimized_flatten_transpose_kernel(
    x_ptr,
    out_ptr,
    n_batch,
    n_seq,
    n_height,
    n_width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Optimized flatten and transpose kernel"""
    # Program identifiers
    batch_id = tl.program_id(0)
    flat_dim_id = tl.program_id(1)
    
    # Calculate offsets
    offset_batch = batch_id * n_seq * n_height * n_width
    offset_flat = flat_dim_id * BLOCK_SIZE_N
    
    # Bounds for flat dimension
    mask_flat = offset_flat + tl.arange(0, BLOCK_SIZE_N) < (n_height * n_width)
    
    # Load input data: [batch, seq, height, width]
    offset_base = offset_batch + offset_flat
    mask_seq = tl.arange(0, n_seq)[:, None]  # Row vector for broadcasting
    
    x = tl.load(x_ptr + offset_base + mask_seq * n_height * n_width, mask=mask_seq, other=0.0)
    
    # Output is transposed: [batch, seq*height*width] -> [batch, seq, height*width] becomes [batch, height*width, seq]
    # We just need to move the data from [batch, seq, flat] -> [batch, flat, seq]
    output_offset = batch_id * (n_height * n_width) * n_seq + offset_flat * n_seq + mask_seq
    
    # Store result in transposed order
    tl.store(out_ptr + output_offset, x, mask=mask_seq and mask_flat[0])

@torch.fx.wrap
def optimized_flatten_transpose(x):
    """Optimized flatten and transpose function"""
    # Get tensor shapes
    n_batch, n_seq, n_height, n_width = x.shape
    n_flat = n_height * n_width
    
    # Create output tensor: [n_batch, n_flat, n_seq]
    out = torch.empty((n_batch, n_flat, n_seq), dtype=x.dtype, device=x.device)
    
    # Configure kernel parameters
    BLOCK_SIZE_N = min(1024, n_flat)
    flat_blocks = triton.cdiv(n_flat, BLOCK_SIZE_N)
    
    # Launch kernel with 2D grid: (batch_size, flat_blocks)
    grid = (n_batch, flat_blocks)
    
    # Launch kernel
    optimized_flatten_transpose_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_batch=n_batch,
        n_seq=n_seq,
        n_height=n_height,
        n_width=n_width,
        BLOCK_SIZE_M=BLOCK_SIZE_N,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return out

def replacement_func():
    """Return the optimized function"""
    return optimized_flatten_transpose