import torch
import triton
import triton.language as tl

def pattern(tmp_9, in_1, in_0):
    # LayerNorm operation with weight and bias
    tmp_10 = torch.nn.functional.layer_norm(tmp_9, (768,), in_1, in_0, 1e-05)
    return tmp_10

def replacement_args(tmp_9, in_1, in_0):
    return (tmp_9, in_1, in_0)

# Triton kernel for optimized LayerNorm
@triton.jit
def layer_norm_kernel(
    x_ptr,           # Input tensor: [batch, seq, features]
    gamma_ptr,       # Weight: [features]
    beta_ptr,        # Bias: [features]
    out_ptr,         # Output: [batch, seq, features]
    n_batch,         # Batch size
    n_seq,           # Sequence length
    n_features,      # Feature dimension
    eps,             # Epsilon for numerical stability
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles a batch x sequence position
    pid = tl.program_id(0)
    
    # Calculate block offsets
    m_offset = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    batch_mask = m_offset // n_seq < n_batch
    seq_mask = m_offset % n_seq < n_seq
    feature_mask = n_offset < n_features
    
    # Combined mask
    mask = batch_mask[:, None] & seq_mask[:, None] & feature_mask[None, None, :]
    
    # Calculate batch and sequence indices
    batch_idx = m_offset // n_seq
    seq_idx = m_offset % n_seq
    
    # Strided offset for feature dimension
    feature_stride = n_seq * n_batch
    x_ptrs = x_ptr + (batch_idx[:, None, None] * feature_stride + seq_idx[None, :, None] * n_features + n_offset[None, None, :])
    
    # Load input data
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Load gamma and beta parameters
    gamma = tl.load(gamma_ptr + n_offset[None, None, :], mask=feature_mask[None, None, :], other=1.0)
    beta = tl.load(beta_ptr + n_offset[None, None, :], mask=feature_mask[None, None, :], other=0.0)
    
    # Compute mean
    local_mean = tl.sum(x, axis=2) / n_features
    
    # Compute variance: E[X^2] - (E[X])^2
    x_centered = x - local_mean[:, :, None]
    x_var = tl.sum(x_centered * x_centered, axis=2) / n_features
    
    # Normalize: (x - mean) / sqrt(var + epsilon)
    x_normalized = x_centered / tl.sqrt(x_var[:, :, None] + eps)
    
    # Apply scale and shift: gamma * normalized + beta
    out = gamma * x_normalized + beta
    
    # Store output
    out_ptrs = out_ptr + (batch_idx[:, None, None] * feature_stride + seq_idx[None, :, None] * n_features + n_offset[None, None, :])
    tl.store(out_ptrs, out, mask=mask)

# Alternative kernel for better performance with small batch size (like our case: batch=1)
@triton.jit
def layer_norm_kernel_batch1(
    x_ptr,           # Input tensor: [1, seq, features]
    gamma_ptr,       # Weight: [features]
    beta_ptr,        # Bias: [features]
    out_ptr,         # Output: [1, seq, features]
    n_seq,           # Sequence length
    n_features,      # Feature dimension
    eps,             # Epsilon for numerical stability
    BLOCK_SIZE_N: tl.constexpr,  # Sequence blocks
    BLOCK_SIZE_M: tl.constexpr,  # Feature blocks
):
    # Each program handles a sequence position x feature block
    pid = tl.program_id(0)
    
    # Calculate block offsets
    seq_offset = (pid // ((n_features + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    feature_offset = (pid % ((n_features + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # Create masks
    seq_mask = seq_offset < n_seq
    feature_mask = feature_offset < n_features
    
    # Combined mask
    mask = seq_mask[:, None] & feature_mask[None, :]
    
    # Memory pointers (we know batch = 1, so simplified offset calculation)
    stride_seq = n_features
    x_ptrs = x_ptr + (seq_offset[:, None] * stride_seq + feature_offset[None, :])
    out_ptrs = out_ptr + (seq_offset[:, None] * stride_seq + feature_offset[None, :])
    
    # Load input data
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Load gamma and beta parameters
    gamma = tl.load(gamma_ptr + feature_offset[None, :], mask=feature_mask[None, :], other=1.0)
    beta = tl.load(beta_ptr + feature_offset[None, :], mask=feature_mask[None, :], other=0.0)
    
    # Compute mean across feature dimension for each sequence position
    local_mean = tl.sum(x, axis=1) / n_features
    local_mean = local_mean[:, None]  # Reshape for broadcasting
    
    # Compute variance
    x_centered = x - local_mean
    x_var = tl.sum(x_centered * x_centered, axis=1) / n_features
    x_var = x_var[:, None]  # Reshape for broadcasting
    
    # Normalize and apply scale/shift
    x_normalized = x_centered / tl.sqrt(x_var + eps)
    out = gamma * x_normalized + beta
    
    # Store output
    tl.store(out_ptrs, out, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(tmp_9, in_1, in_0):
    batch_size, n_seq, n_features = tmp_9.shape
    
    # For small batch sizes (like batch=1), use optimized kernel
    if batch_size == 1:
        # Output shape remains [1, n_seq, n_features]
        output = torch.empty_like(tmp_9)
        
        # Block sizes optimized for our specific shapes
        BLOCK_SIZE_N = 64   # Sequence positions per block
        BLOCK_SIZE_M = 128  # Features per block
        
        # Calculate grid size
        num_blocks_seq = (n_seq + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        num_blocks_features = (n_features + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        total_blocks = num_blocks_seq * num_blocks_features
        
        # Launch kernel optimized for batch=1
        layer_norm_kernel_batch1[(total_blocks,)](
            tmp_9,
            in_1,
            in_0,
            output,
            n_seq,
            n_features,
            1e-05,  # epsilon
            BLOCK_SIZE_N,
            BLOCK_SIZE_M
        )
    else:
        # Fallback for larger batches
        output = torch.empty_like(tmp_9)
        
        BLOCK_SIZE_M = 32   # Batch blocks
        BLOCK_SIZE_N = 128  # Sequence blocks
        
        num_blocks_batch = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        num_blocks_seq = (n_seq + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        
        layer_norm_kernel[(num_blocks_batch * num_blocks_seq,)](
            tmp_9,
            in_1,
            in_0,
            output,
            batch_size,
            n_seq,
            n_features,
            1e-05,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N
        )
    
    return output

def replacement_func():
    return optimized_layer_norm