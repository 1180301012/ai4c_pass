import torch
import triton
import triton.language as tl

# Pattern matching: complete computation chain
def pattern(x, y):
    # Step 1: Scalar multiplication with hardcoded 0.0625
    scaled = 0.0625 * x
    # Step 2: Softmax
    softmax = torch.nn.functional.softmax(scaled, dim=-1)
    # Step 3: Matrix multiplication
    matmul_out = torch.matmul(softmax, y)
    # Step 4: Transpose
    transposed = matmul_out.permute(0, 2, 1)
    return transposed

# Argument extraction
def replacement_args(x, y):
    return (x, y)

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
    seq_dim = pid % seq_len
    dim_pid = (pid // seq_len) % dim_size
    batch_pid = pid // (seq_len * dim_size)
    
    if batch_pid >= batch_size:
        return
    
    # Accumulator for the final output
    accumulator = tl.zeros((BLOCK_SIZE_FEAT, BLOCK_SIZE_DIM), dtype=tl.float32)
    
    # Load x data and compute softmax
    x_ptrs = x_ptr + (batch_pid * seq_len * features + seq_dim * features + tl.arange(0, BLOCK_SIZE_FEAT))
    x_mask = tl.arange(0, BLOCK_SIZE_FEAT) < features
    x_data = tl.load(x_ptrs, mask=x_mask, other=0.0)
    
    # Apply scaling and softmax
    scaled_data = 0.0625 * x_data
    max_val = tl.max(scaled_data)
    exp_scaled = tl.exp(scaled_data - max_val)
    sum_exp = tl.sum(exp_scaled)
    softmax_weights = exp_scaled / sum_exp
    
    # Load y data and compute weighted sum
    for feat_idx in range(0, features, BLOCK_SIZE_FEAT):
        y_ptrs = y_ptr + (batch_pid * features * dim_size + feat_idx * dim_size + tl.arange(0, BLOCK_SIZE_DIM))
        y_mask = tl.arange(0, BLOCK_SIZE_DIM) < dim_size
        y_data = tl.load(y_ptrs, mask=y_mask, other=0.0)
        
        # Weighted sum contribution
        weighted_contrib = softmax_weights[feat_idx] * y_data if feat_idx < BLOCK_SIZE_FEAT else accumulator
        accumulator += weighted_contrib
    
    # Store result to transposed output: [batch, dim, seq]
    out_ptrs = out_ptr + (batch_pid * dim_size * seq_len + dim_pid * seq_len + seq_dim)
    out_mask = (dim_pid < dim_size) & (seq_dim < seq_len)
    tl.store(out_ptrs, accumulator[0], mask=out_mask)

@torch.fx.wrap
def triton_end_to_end_fusion(x, y):
    batch_size, seq_len, features = x.shape
    batch_size_y, features_y, dim_size = y.shape
    
    assert batch_size == batch_size_y
    assert features == features_y
    
    BLOCK_SIZE_FEAT = 32
    BLOCK_SIZE_DIM = 64
    
    # Calculate grid size
    total_elements = batch_size * seq_len * dim_size
    grid = (total_elements + 255) // 256  # Round up to nearest 256
    
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
        BLOCK_SIZE_SEQ=1,
        BLOCK_SIZE_FEAT=BLOCK_SIZE_FEAT,
        BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
    )
    
    return out

def replacement_func():
    return triton_end_to_end_fusion