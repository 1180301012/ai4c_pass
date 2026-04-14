import torch
import triton
import triton.language as tl

def pattern(in_0, in_2):
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_17 = in_0 * tmp_16
    return tmp_17

def replacement_args(in_0, in_2):
    return (in_0, in_2)

@triton.jit
def rms_norm_kernel(
    inputs_ptr, weights_ptr, 
    output_ptr, 
    n_elements, feature_dim, batch_size, seq_len,
    eps: tl.constexpr, BLOCK_SIZE_ROWS: tl.constexpr, BLOCK_SIZE_COLS: tl.constexpr
):
    # Each program handles one feature dimension
    row_id = tl.program_id(0)
    col_id = tl.program_id(1)
    
    # Load weights for this feature dimension
    weight = tl.load(weights_ptr + row_id)
    
    # Compute mean of squares for this row across all sequences in the batch
    sum_sq = 0.0
    for seq_idx in range(batch_size):
        for seq_pos_idx in range(seq_len):
            element_idx = seq_idx * seq_len * feature_dim + seq_pos_idx * feature_dim + row_id
            if element_idx < n_elements:
                val = tl.load(inputs_ptr + element_idx)
                sum_sq += val * val
    
    # Compute mean and rsqrt
    mean_sq = sum_sq / (batch_size * seq_len)
    inv_std = 1.0 / tl.sqrt(mean_sq + eps)
    
    # Apply normalization for this row
    for seq_idx in range(batch_size):
        for seq_pos_idx in range(seq_len):
            element_idx = seq_idx * seq_len * feature_dim + seq_pos_idx * feature_dim + row_id
            if element_idx < n_elements:
                val = tl.load(inputs_ptr + element_idx)
                normalized_val = val * inv_std
                result = normalized_val * weight
                tl.store(output_ptr + element_idx, result)



@torch.fx.wrap
def optimized_rms_norm(in_0, in_2):
    # Get input dimensions
    batch_size, seq_len, feature_dim = in_2.shape
    n_elements = in_2.numel()
    
    BLOCK_SIZE_ROWS = 64  # Process multiple features per thread block for better utilization
    BLOCK_SIZE_COLS = 128  # Process multiple sequence positions per thread
    grid = (feature_dim // BLOCK_SIZE_ROWS + 1, batch_size * seq_len // BLOCK_SIZE_COLS + 1)
    
    # Allocate output - should have same shape as in_0 [2048]
    output = torch.empty_like(in_0)
    
    # Convert in_2 to float32 for computation
    inputs_float32 = in_2.to(torch.float32)
    
    # Call the kernel
    rms_norm_kernel[grid](
        inputs_ptr=inputs_float32,
        weights_ptr=in_0,  # in_0 acts as the scaling weight
        output_ptr=output,
        n_elements=inputs_float32.numel(),
        feature_dim=feature_dim,
        batch_size=batch_size,
        seq_len=seq_len,
        eps=1e-06,
        BLOCK_SIZE_ROWS=BLOCK_SIZE_ROWS,
        BLOCK_SIZE_COLS=BLOCK_SIZE_COLS
    )
    
    return output

def replacement_func():
    return optimized_rms_norm