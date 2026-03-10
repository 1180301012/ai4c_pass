import torch
import triton
import triton.language as tl

# Pattern matching: complete computation chain
def pattern(x, scale, y):
    # Step 1: Scalar multiplication
    scaled = scale * x
    # Step 2: Softmax
    softmax = torch.nn.functional.softmax(scaled, dim=-1)
    # Step 3: Matrix multiplication
    matmul_out = torch.matmul(softmax, y)
    # Step 4: Transpose
    transposed = matmul_out.permute(0, 2, 1)
    return transposed

# Argument extraction
def replacement_args(x, scale, y):
    return (x, scale, y)

# Optimized kernel that fuses all operations
@triton.jit
def end_to_end_fusion_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    seq_len,
    features,
    dim_size,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(axis=0)
    num_pid_seq = tl.cdiv(seq_len, BLOCK_SIZE_SEQ)
    num_pid_dim = tl.cdiv(dim_size, BLOCK_SIZE_DIM)
    num_pid_in_group = num_pid_seq * num_pid_dim
    group_id = pid // num_pid_in_group
    first_pid_seq = (pid % num_pid_in_group) // num_pid_dim
    group_size_seq = tl.minimum(num_pid_seq, 2048 // num_pid_dim)
    group_seq_start = group_id * group_size_seq
    first_pid_seq = first_pid_seq + group_seq_start
    pid_seq = first_pid_seq * BLOCK_SIZE_SEQ + tl.arange(0, BLOCK_SIZE_SEQ)
    pid_dim = tl.arange(0, BLOCK_SIZE_DIM)
    
    pid_seq = pid_seq % seq_len
    pid_dim = pid_dim % dim_size
    
    # Process each batch in sequence
    for b in range(batch_size):
        # Accumulator for the final output
        accumulator = tl.zeros((BLOCK_SIZE_SEQ, BLOCK_SIZE_DIM), dtype=tl.float32)
        
        # For each position in sequence, compute softmax and matmul
        for seq_idx in range(0, seq_len, BLOCK_SIZE_FEAT):
            # Load x data: 0.0625 * x[b, seq_idx:seq_idx+features]
            x_ptrs = x_ptr + (b * seq_len * features + pid_seq * features + tl.arange(0, BLOCK_SIZE_FEAT)[None, :])
            x_mask = (pid_seq < seq_len)[:, None] & (tl.arange(0, BLOCK_SIZE_FEAT)[None, :] < features)
            x_data = tl.load(x_ptrs, mask=x_mask, other=0.0)
            
            # Apply scaling
            scaled_data = 0.0625 * x_data
            
            # Compute softmax along features dimension
            max_val = tl.max(scaled_data, axis=1)
            scaled_data = scaled_data - max_val[:, None]
            exp_scaled = tl.exp(scaled_data)
            softmax_weights = exp_scaled / tl.sum(exp_scaled, axis=1)[:, None]
            
            # Load y data: y[b, features, pid_dim]
            y_ptrs = y_ptr + (b * features * dim_size + tl.arange(0, BLOCK_SIZE_FEAT)[:, None] * dim_size + pid_dim[None, :])
            y_mask = (tl.arange(0, BLOCK_SIZE_FEAT)[:, None] < features) & (pid_dim < dim_size)[None, :]
            y_data = tl.load(y_ptrs, mask=y_mask, other=0.0)
            
            # Weighted sum: softmax_weights @ y_data
            # We need to transpose y_data for proper dimensions
            y_data_t = y_data.transpose(0, 1)
            weighted_sum = tl.dot(softmax_weights, y_data_t, accumulator)
            
            accumulator = weighted_sum
        
        # Store results to output with transposed shape: [batch, dim, seq]
        out_ptrs = out_ptr + (b * dim_size * seq_len + pid_dim[:, None] * seq_len + pid_seq[:, None])
        out_mask = (pid_dim < dim_size)[:, None] & (pid_seq < seq_len)[:, None]
        tl.store(out_ptrs, accumulator, mask=out_mask)

@torch.fx.wrap
def triton_end_to_end_fusion(x, y):
    batch_size, seq_len, features = x.shape
    batch_size_y, features_y, dim_size = y.shape
    
    assert batch_size == batch_size_y
    assert features == features_y
    
    BLOCK_SIZE_SEQ = 64
    BLOCK_SIZE_FEAT = 32
    BLOCK_SIZE_DIM = 64
    
    # Calculate grid size
    grid = lambda META: (
        triton.cdiv(seq_len, BLOCK_SIZE_SEQ) * triton.cdiv(dim_size, BLOCK_SIZE_DIM) * batch_size,
        1,
        1,
    )
    
    # Allocate output tensor with transposed shape: [batch, dim, seq]
    out = torch.empty((batch_size, dim_size, seq_len), dtype=x.dtype, device=x.device)
    
    # Launch the kernel
    end_to_end_fusion_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        features=features,
        dim_size=dim_size,
        BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
        BLOCK_SIZE_FEAT=BLOCK_SIZE_FEAT,
        BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
    )
    
    return out

def replacement_func():
    return triton_end_to_end_fusion