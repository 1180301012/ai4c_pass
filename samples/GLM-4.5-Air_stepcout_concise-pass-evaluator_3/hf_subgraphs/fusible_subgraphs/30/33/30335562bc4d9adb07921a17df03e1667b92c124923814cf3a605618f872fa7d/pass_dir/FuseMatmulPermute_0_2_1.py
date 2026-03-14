import torch
import triton
import triton.language as tl

def pattern(a, b):
    """Pattern to match matmul followed by permute(0, 2, 1)"""
    tmp_2 = torch.matmul(a, b)
    tmp_3 = tmp_2.permute(0, 2, 1)
    return tmp_3

def replacement_args(a, b):
    """Extract arguments for replacement"""
    return (a, b)

@triton.jit
def fused_matmul_permute_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    batch_size,
    a_m,
    a_k,
    b_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Fused kernel that performs matmul and transposes result directly to [batch, n, m]"""
    # Program ID for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets for output [batch, n, m] shape
    batch_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Calculate bounds
    m_end = (batch_offset + BLOCK_SIZE_M) // batch_size
    if m_end <= a_m and pid_n * BLOCK_SIZE_N < b_n:
        # Process one batch at a time within this block
        for batch_idx in range(batch_offset, min(batch_offset + BLOCK_SIZE_M, batch_size * a_m), a_m):
            # Final output indices directly in [batch, n, m] shape
            batch_id = batch_idx // a_m
            
            # For each n dimension in the output
            for n_local in range(BLOCK_SIZE_N):
                n_global = n_offset + n_local
                if n_global < b_n:
                    acc = 0.0
                    # Accumulate over k dimension
                    for k in range(a_k):
                        # Load from a: [batch, m, k] -> convert to [batch, k, m] indexing
                        a_offset = batch_idx + k * a_m
                        a_val = tl.load(a_ptr + a_offset + tl.arange(0, min(BLOCK_SIZE_M, a_m - (batch_idx % a_m))), 
                                       mask=tl.arange(0, min(BLOCK_SIZE_M, a_m - (batch_idx % a_m))) < (a_m - (batch_idx % a_m)), 
                                       other=0.0)
                        
                        # Load from b: [batch, k, n] -> convert to [batch, n, k] indexing
                        b_offset = batch_idx + n_global * a_k + k
                        b_val = tl.load(b_ptr + b_offset)
                        
                        # Accumulate
                        acc += a_val * b_val
                    
                    # Store result directly in [batch, n, m] shape
                    # We need to handle the transpose carefully
                    output_offset = batch_id * (b_n * a_m) + n_global * a_m + (batch_idx % a_m)
                    tl.store(out_ptr + output_offset, acc)

@torch.fx.wrap  
def fused_matmul_permute(a, b):
    """Apply fused matmul + direct transpose to [batch, n, m] shape"""
    batch, m, k = a.shape  # a: [batch, m, k]
    batch_b, k2, n = b.shape  # b: [batch, k, n]
    assert k == k2, "Inner dimensions must match for matrix multiplication"
    
    # Output will be [batch, n, m] - transposed from normal matmul result [batch, m, n]
    out_shape = (batch, n, m)
    out = torch.empty(out_shape, dtype=a.dtype, device=a.device)
    
    # Use block sizes appropriate for the tensor dimensions
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32  
    BLOCK_SIZE_K = 64
    
    # Calculate grid dimensions
    grid_m = (batch * m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    fused_matmul_permute_kernel[grid_m, grid_n](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        batch_size=batch,
        a_m=m,
        a_k=k,
        b_n=n,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

def replacement_func():
    """Return the fused function"""
    return fused_matmul_permute