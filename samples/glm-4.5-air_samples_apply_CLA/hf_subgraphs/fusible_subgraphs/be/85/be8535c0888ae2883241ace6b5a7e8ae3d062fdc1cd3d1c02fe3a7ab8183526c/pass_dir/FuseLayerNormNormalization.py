import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern: Layer normalization normalization step
    Converts to float32, squares, computes mean, adds epsilon, rsqrt, and multiplies
    """
    t1 = x.to(torch.float32)
    t2 = t1.pow(2)
    t3 = t2.mean(-1, keepdim=True)
    t4 = t3 + 1e-06
    t5 = torch.rsqrt(t4)
    out = x * t5
    return out

def replacement_args(x):
    """Extract arguments for the kernel: input tensor"""
    return (x,)

@triton.jit
def compute_sums_kernel(
    x_ptr,
    sums_ptr,
    n_batch,
    n_seq,
    n_hidden,
    eps: tl.constexpr,
):
    """Kernel to compute sum of squares for each hidden dimension in parallel"""
    # Each program handles one (batch, seq, hidden) combination
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    hidden_idx = tl.program_id(2)
    
    # Compute pointer for this specific location
    ptr = x_ptr + batch_idx * n_seq * n_hidden + seq_idx * n_hidden + hidden_idx
    
    # Load input value
    x_val = tl.load(ptr)
    x_float = x_val.to(tl.float32)
    
    # Store square value (we'll sum these elsewhere)
    square = x_float * x_float
    offset = batch_idx * n_seq * n_hidden + seq_idx * n_hidden + hidden_idx
    tl.store(sums_ptr + offset, square)

@triton.jit
def sum_squares_kernel(
    squares_ptr,
    mean_squares_ptr,
    n_batch,
    n_seq,
    n_hidden,
    eps: tl.constexpr,
):
    """Optimized kernel to sum squares across batch and sequence dimensions for each hidden"""
    # Each program handles one hidden dimension
    hidden_idx = tl.program_id(0)
    
    # Sum squares across all batch and sequence combinations for this hidden dimension
    sum_squares = 0.0
    
    # Process in larger blocks for better memory access patterns
    batch_seq_block_size = 16
    
    for batch_seq_block in range(0, n_batch * n_seq, batch_seq_block_size):
        block_remaining = min(batch_seq_block_size, n_batch * n_seq - batch_seq_block)
        
        # Accumulate a block of squares
        block_sum = 0.0
        for i in range(block_remaining):
            block_idx = batch_seq_block + i
            batch_idx = block_idx // n_seq
            seq_idx = block_idx % n_seq
            offset = batch_idx * n_seq * n_hidden + seq_idx * n_hidden + hidden_idx
            square = tl.load(squares_ptr + offset)
            block_sum += square
        
        sum_squares += block_sum
    
    # Compute mean
    mean_squared = sum_squares / (n_batch * n_seq)
    tl.store(mean_squares_ptr + hidden_idx, mean_squared)

@triton.jit 
def variance_scaling_apply_kernel(
    x_ptr,
    mean_squares_ptr,
    out_ptr,
    n_batch,
    n_seq,
    n_hidden,
    eps: tl.constexpr,
):
    """Apply variance scaling using precomputed mean squares"""
    # Each program handles one (batch, seq, hidden) combination
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1) 
    hidden_idx = tl.program_id(2)
    
    # Compute pointer for this specific location
    ptr = x_ptr + batch_idx * n_seq * n_hidden + seq_idx * n_hidden + hidden_idx
    
    # Load input value
    x_val = tl.load(ptr)
    x_float = x_val.to(tl.float32)
    
    # Load mean square for this hidden dimension
    mean_squared = tl.load(mean_squares_ptr + hidden_idx)
    
    # Compute reciprocal square root with epsilon
    inv_std = 1.0 / tl.sqrt(mean_squared + eps)
    
    # Apply scaling
    scaled = x_float * inv_std
    
    # Store result
    out_offset = batch_idx * n_seq * n_hidden + seq_idx * n_hidden + hidden_idx
    tl.store(out_ptr + out_offset, scaled)

@torch.fx.wrap
def fused_layer_norm(x):
    """High-performance fused variance scaling"""
    if x.dim() != 3:
        raise ValueError("Expected 3D tensor [batch, seq, hidden]")
    
    n_batch, n_seq, n_hidden = x.shape
    
    # Step 1: Compute squares in parallel using Triton
    squares = torch.empty((n_batch, n_seq, n_hidden), dtype=torch.float32, device=x.device)
    grid_size = (n_batch, n_seq, n_hidden)
    
    compute_sums_kernel[grid_size](
        x_ptr=x,
        sums_ptr=squares,
        n_batch=n_batch,
        n_seq=n_seq,
        n_hidden=n_hidden,
        eps=1e-06,
    )
    
    # Step 2: Compute mean squares using Triton kernel
    mean_squares = torch.empty(n_hidden, dtype=torch.float32, device=x.device)
    mean_grid_size = (n_hidden,)
    
    sum_squares_kernel[mean_grid_size](
        squares_ptr=squares,
        mean_squares_ptr=mean_squares,
        n_batch=n_batch,
        n_seq=n_seq,
        n_hidden=n_hidden,
        eps=1e-06,
    )
    
    # Step 3: Apply scaling in parallel
    out = torch.empty((n_batch, n_seq, n_hidden), dtype=torch.float32, device=x.device)
    
    # Launch Triton kernel for final scaling step
    variance_scaling_apply_kernel[grid_size](
        x_ptr=x,
        mean_squares_ptr=mean_squares,
        out_ptr=out,
        n_batch=n_batch,
        n_seq=n_seq,
        n_hidden=n_hidden,
        eps=1e-06,
    )
    
    return out

def replacement_func():
    """Return the fused layer normalization function"""
    return fused_layer_norm