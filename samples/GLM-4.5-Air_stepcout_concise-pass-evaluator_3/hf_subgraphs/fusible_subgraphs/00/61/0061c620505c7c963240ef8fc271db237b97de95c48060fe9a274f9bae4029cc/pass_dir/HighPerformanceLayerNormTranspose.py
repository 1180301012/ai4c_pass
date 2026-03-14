import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    tmp_2 = torch.nn.functional.layer_norm(in_2, (768,), in_1, in_0, 1e-05)
    tmp_3 = tmp_2.transpose(-1, -2)
    return tmp_3

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def high_performance_layer_norm_transpose_kernel(
    x_ptr, 
    weight_ptr,
    bias_ptr,
    out_ptr,
    num_features,
    seq_len,
    batch_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # sequence dimension
    
    # Compute block range for this program
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask for valid indices
    m_mask = m_offsets < batch_size
    n_mask = n_offsets < seq_len
    
    # Create meshgrid for iteration
    m_offsets, n_offsets = tl.meshgrid(m_offsets, n_offsets)
    mask = m_mask[:, None] & n_mask[None, :]
    
    # Only proceed if we have valid data
    if tl.sum(mask) == 0:
        return
    
    # Initialize mean and variance accumulators
    mean_sum = 0.0
    variance_sum = 0.0
    
    # Number of elements contributing to this variance
    num_elements = tl.sum(mask)
    
    # Compute mean and variance for this block
    # Load input tensor values for all positions in this block
    x_offsets = m_offsets * seq_len * num_features + n_offsets * num_features
    x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    
    # Compute sum for mean
    mean_sum = tl.sum(x_vals)
    mean = mean_sum / num_elements
    
    # Compute sum of squares for variance
    variance_sum = tl.sum((x_vals - mean) * (x_vals - mean))
    variance = tl.sqrt(variance_sum / num_elements + 1e-05)
    
    # Load weights and biases (broadcast across block)
    weight_vals = tl.load(weight_ptr, mask=mask, other=0.0)
    bias_vals = tl.load(bias_ptr, mask=mask, other=0.0)
    
    # Apply layer normalization
    normalized = (x_vals - mean) / variance * weight_vals + bias_vals
    
    # Compute transposed output offsets: [batch, features, seq]
    output_offsets = m_offsets * num_features * seq_len + tl.arange(num_features)[None, :] * seq_len + n_offsets
    feature_mask = tl.arange(num_features)[None, :] < num_features
    output_mask = mask & feature_mask
    
    # Store transposed results
    tl.store(out_ptr + output_offsets, normalized, mask=output_mask)

@torch.fx.wrap
def high_performance_layer_norm_transpose(x, weight, bias):
    batch_size, seq_len, num_features = x.shape
    
    # Choose optimal block sizes for GPU occupancy
    BLOCK_SIZE_M = 4   # batch dimension block
    BLOCK_SIZE_N = 32  # sequence dimension block
    
    # Compute grid size
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor (transposed: [batch, features, seq])
    out = torch.empty((batch_size, num_features, seq_len), dtype=x.dtype, device=x.device)
    
    # Launch optimized kernel
    high_performance_layer_norm_transpose_kernel[(grid_m, grid_n)](
        x,
        weight,
        bias,
        out,
        num_features,
        seq_len,
        batch_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    return high_performance_layer_norm_transpose