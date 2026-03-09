import torch
import triton
import triton.language as tl

def pattern(in_2):
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    return tmp_16

@triton.jit
def fused_normalization_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one row (token)
    row = tl.program_id(0)
    col_offset = tl.program_id(1) * BLOCK_SIZE_N
    cols = col_offset + tl.arange(0, BLOCK_SIZE_N)
    
    # Mask for valid columns
    mask = cols < hidden_dim
    
    # Load the entire row (token embedding)
    x = tl.load(x_ptr + row * hidden_dim + cols, mask=mask, other=0.0)
    
    # Square the values
    x_squared = x * x
    
    # Compute mean along hidden dimension
    # We'll use a parallel reduction for the mean computation
    block_squared = tl.sum(x_squared, axis=0)
    
    # Here we need to handle the reduction across all rows and compute global mean
    # For simplicity, we'll compute mean per row first then reduce
    local_mean = block_squared / hidden_dim
    
    # Add epsilon
    mean_with_epsilon = local_mean + 1e-06
    
    # Compute reciprocal square root
    inv_std = tl.rsqrt(mean_with_epsilon)
    
    # Normalize: x * inv_std
    x_normalized = x * inv_std
    
    # Convert back to bfloat16 and store
    tl.store(out_ptr + row * hidden_dim + cols, x_normalized.to(tl.bfloat16), mask=mask)

@triton.jit
def fused_normalization_kernel_parallel(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles multiple elements
    pid = tl.program_id(0)
    row = pid // (hidden_dim // BLOCK_SIZE)
    col = (pid % (hidden_dim // BLOCK_SIZE)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = col < hidden_dim
    
    # Load the current row's data
    x = tl.load(x_ptr + row * hidden_dim + col, mask=mask, other=0.0)
    
    # Compute squared values for the row (but limit to our block)
    x_squared = x * x
    
    # For this block, compute partial sum
    partial_sum = tl.sum(x_squared, axis=0)
    
    # All blocks in the row need to participate in the final reduction
    # For now, we'll handle in a simplified way - each row processes independently
    # This is not optimal for the mean computation but correct for the algorithm
    
    # Each thread should compute the mean for its assigned row
    # Since we're processing segments, we need to sum across all segments for each row
    # This is a limitation - we can't easily do cross-thread reduction in this pattern
    # Let's use a different approach

# Alternative approach with better memory access patterns
@torch.fx.wrap  
def fused_normalization_ops(in_2):
    batch_size, seq_len, hidden_dim = in_2.shape
    
    out = torch.empty((batch_size, seq_len, hidden_dim), dtype=torch.bfloat16, device=in_2.device)
    
    # We'll process each token independently for simplicity in this implementation
    # A more sophisticated implementation would use shared memory for reduction
    BLOCK_SIZE = 128
    
    # For each token (row), compute normalization
    for r in range(batch_size * seq_len):
        # Load the row as float32
        x = in_2[r].to(torch.float32)
        
        # Compute mean of squares
        x_squared = x * x
        mean_squared = torch.sum(x_squared) / hidden_dim
        
        # Add epsilon and compute rsqrt using torch
        inv_std = 1.0 / torch.sqrt(mean_squared + 1e-06)
        
        # Normalize
        x_normalized = x * inv_std
        
        # Store result
        out[r] = x_normalized.to(torch.bfloat16)
    
    return out

# Correct Triton implementation for layer normalization
@triton.jit
def fused_normalization_kernel_correct(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    REDUCTION_BLOCK_SIZE: tl.constexpr,
    COMPUTE_BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one token (row) for the reduction
    token_id = tl.program_id(0)
    
    mask = token_id < batch_size * seq_len
    batch_id = token_id // seq_len
    seq_id = token_id % seq_len
    
    # Initialize sum to 0
    sum_squared = 0.0
    
    # Step 1: Compute sum of squares across the hidden dimension
    for start in range(0, hidden_dim, REDUCTION_BLOCK_SIZE):
        end = min(start + REDUCTION_BLOCK_SIZE, hidden_dim)
        offsets = start + tl.arange(0, end - start)
        
        # Load block
        x_block = tl.load(x_ptr + token_id * hidden_dim + offsets, 
                         mask=offsets < hidden_dim, other=0.0)
        x_block_float = x_block.to(tl.float32)
        
        # Sum squares
        block_squared = tl.sum(x_block_float * x_block_float)
        sum_squared += block_squared
    
    # Compute mean and reciprocal standard deviation
    mean_squared = sum_squared / hidden_dim
    inv_std = tl.rsqrt(mean_squared + 1e-06)
    
    # Step 2: Normalize the entire token
    for start in range(0, hidden_dim, COMPUTE_BLOCK_SIZE):
        end = min(start + COMPUTE_BLOCK_SIZE, hidden_dim)
        offsets = start + tl.arange(0, end - start)
        
        # Load block
        x_block = tl.load(x_ptr + token_id * hidden_dim + offsets,
                         mask=offsets < hidden_dim, other=0.0)
        x_block_float = x_block.to(tl.float32)
        
        # Normalize
        x_normalized = x_block_float * inv_std
        
        # Store as bfloat16
        tl.store(out_ptr + token_id * hidden_dim + offsets, 
                x_normalized.to(tl.bfloat16), mask=offsets < hidden_dim)

@torch.fx.wrap
def fused_normalization_ops_correct(in_2):
    batch_size, seq_len, hidden_dim = in_2.shape
    out = torch.empty((batch_size, seq_len, hidden_dim), dtype=torch.bfloat16, device=in_2.device)
    
    REDUCTION_BLOCK_SIZE = 256
    COMPUTE_BLOCK_SIZE = 256
    num_tokens = batch_size * seq_len
    
    fused_normalization_kernel_correct[(num_tokens,)](
        in_2,
        out,
        batch_size,
        seq_len,
        hidden_dim,
        REDUCTION_BLOCK_SIZE,
        COMPUTE_BLOCK_SIZE
    )
    
    return out

def replacement_args(in_2):
    return (in_2,)

def replacement_func():
    return fused_normalization_ops_correct