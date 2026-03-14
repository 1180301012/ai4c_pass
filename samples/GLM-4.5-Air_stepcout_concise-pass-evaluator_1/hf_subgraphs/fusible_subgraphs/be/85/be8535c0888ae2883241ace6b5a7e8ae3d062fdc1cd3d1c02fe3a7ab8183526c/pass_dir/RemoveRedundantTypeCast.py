import torch
import triton
import triton.language as tl

@triton.jit
def fused_layer_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    n_elements,
    n_hidden_dims,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (batch position * sequence)
    pid = tl.program_id(0)
    
    if pid >= n_rows:
        return
    
    # Calculate offsets for this row
    row_offset = pid * n_hidden_dims
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data for this row
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean for this row
    sum_x = tl.sum(x)
    mean_x = sum_x / n_hidden_dims
    
    # Compute variance
    x_centered = x - mean_x
    x2 = x_centered * x_centered
    sum_x2 = tl.sum(x2)
    var_x = sum_x2 / n_hidden_dims
    
    # Add epsilon and compute rsqrt
    eps = 1e-06
    rsqrt_var = tl.rsqrt(var_x + eps)
    
    # Normalize: (x - mean) * rsqrt_var
    normalized = x_centered * rsqrt_var
    
    # Store the normalized result (weight multiplication done in wrapper)
    tl.store(output_ptr + offsets, normalized, mask=mask)

@torch.fx.wrap  
def optimized_layer_norm(tmp_1, in_0):
    batch_size, seq_len, hidden_size = tmp_1.shape
    n_elements = tmp_1.numel()
    n_rows = batch_size * seq_len  # Number of rows (batch * sequence)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Compute normalized result
    normalized_output = torch.empty_like(tmp_1)
    
    fused_layer_norm_kernel[(num_programs,)](
        input_ptr=tmp_1,
        weight_ptr=tmp_1,  # Dummy weight, we handle multiplication outside
        output_ptr=normalized_output,
        n_elements=n_elements,
        n_hidden_dims=hidden_size,
        n_rows=n_rows,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Apply final weight multiplication (equivalent to tmp_10 = in_0 * normalized_output)
    final_output = in_0 * normalized_output
    
    return final_output

def pattern(tmp_1, in_0):
    # Pattern to match the normalization subgraph (tmp_4 through tmp_10)
    tmp_4 = tmp_1.to(torch.float32)
    tmp_5 = tmp_4.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_9 = tmp_1 * tmp_8
    tmp_10 = in_0 * tmp_9
    return tmp_10

def replacement_args(tmp_1, in_0):
    return (tmp_1, in_0)

def replacement_func():
    return optimized_layer_norm