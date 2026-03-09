import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # LayerNorm computation with initial element-wise addition
    tmp_3 = in_3 + in_2
    tmp_4 = tmp_3.float()
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    tmp_9 = tmp_4 - tmp_5
    tmp_12 = tmp_9 / tmp_11
    tmp_13 = tmp_12.to(torch.float32)
    tmp_14 = in_1 * tmp_13
    tmp_15 = tmp_14 + in_0
    return tmp_15

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def compute_mean_kernel(
    input1_ptr,
    input2_ptr,
    mean_ptr,
    batch_size,
    seq_len, 
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program computes mean for one batch x sequence position
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Initialize accumulator
    sum_val = 0.0
    
    # Iterate over hidden dimension
    for hidden_idx in range(0, hidden_size, BLOCK_SIZE):
        # Compute current block bounds
        block_start = hidden_idx
        block_end = min(hidden_idx + BLOCK_SIZE, hidden_size)
        
        # Process current block
        for i in range(block_start, block_end):
            pos = batch_idx * seq_len * hidden_size + seq_idx * hidden_size + i
            val1 = tl.load(input1_ptr + pos)
            val2 = tl.load(input2_ptr + pos)
            sum_val += (val1 + val2)
    
    # Compute mean and store
    mean = sum_val / hidden_size
    mean_pos = batch_idx * seq_len + seq_idx
    tl.store(mean_ptr + mean_pos, mean)

@triton.jit
def compute_variance_kernel(
    input1_ptr,
    input2_ptr,
    mean_ptr,
    variance_ptr,
    batch_size,
    seq_len,
    hidden_size,
    epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program computes variance for one batch x sequence position
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Load mean for this position
    mean_pos = batch_idx * seq_len + seq_idx
    mean = tl.load(mean_ptr + mean_pos)
    
    # Initialize accumulator
    sum_sq = 0.0
    
    # Iterate over hidden dimension
    for hidden_idx in range(0, hidden_size, BLOCK_SIZE):
        # Compute current block bounds
        block_start = hidden_idx
        block_end = min(hidden_idx + BLOCK_SIZE, hidden_size)
        
        # Process current block
        for i in range(block_start, block_end):
            pos = batch_idx * seq_len * hidden_size + seq_idx * hidden_size + i
            val1 = tl.load(input1_ptr + pos)
            val2 = tl.load(input2_ptr + pos)
            val = val1 + val2
            diff = val - mean
            sum_sq += diff * diff
    
    # Compute variance and store
    variance = sum_sq / hidden_size
    std_dev = tl.sqrt(variance + epsilon)
    variance_pos = batch_idx * seq_len + seq_idx
    tl.store(variance_ptr + variance_pos, std_dev)

@triton.jit
def fused_layernorm_kernel(
    input1_ptr,
    input2_ptr,
    mean_ptr,
    variance_ptr,
    bias_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_size,
    epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one element
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    hidden_idx = tl.program_id(2)
    
    # Calculate pointer for this specific position
    pos = batch_idx * seq_len * hidden_size + seq_idx * hidden_size + hidden_idx
    
    # Load input values and perform element-wise addition
    val1 = tl.load(input1_ptr + pos)
    val2 = tl.load(input2_ptr + pos)
    added_val = val1 + val2
    
    # Load mean and std dev for this batch x sequence position
    mean_pos = batch_idx * seq_len + seq_idx
    mean = tl.load(mean_ptr + mean_pos)
    std_dev = tl.load(variance_ptr + mean_pos)
    
    # Load bias and weight parameters for this hidden dimension
    bias = tl.load(bias_ptr + hidden_idx)
    weight = tl.load(weight_ptr + hidden_idx)
    
    # Normalize and apply layer norm
    normalized = (added_val - mean) / std_dev
    result = weight * normalized + bias
    
    # Store the result
    tl.store(output_ptr + pos, result)

@torch.fx.wrap
def fused_layernorm_forward(in_0, in_1, in_2, in_3):
    # Input shapes
    bias_shape = in_0.shape  # [768]
    weight_shape = in_1.shape  # [768]  
    batch_size, seq_len, hidden_size = in_2.shape  # [batch, seq, hidden]
    
    # Create intermediate buffers
    mean_buffer = torch.empty(batch_size * seq_len, dtype=torch.float32, device=in_2.device)
    variance_buffer = torch.empty(batch_size * seq_len, dtype=torch.float32, device=in_2.device)
    
    # Output shape matches input shape
    output_shape = in_2.shape
    output = torch.empty(output_shape, dtype=torch.float32, device=in_2.device)
    
    BLOCK_SIZE = 256  # Optimal block size for GPU
    
    # Step 1: Compute mean across hidden dimension (in_2 + in_3)
    mean_grid = (batch_size, seq_len)
    compute_mean_kernel[mean_grid](
        in_2,
        in_3,
        mean_buffer,
        batch_size,
        seq_len,
        hidden_size,
        BLOCK_SIZE
    )
    
    # Step 2: Compute variance across hidden dimension 
    compute_variance_kernel[mean_grid](
        in_2,
        in_3, 
        mean_buffer,
        variance_buffer,
        batch_size,
        seq_len,
        hidden_size,
        1e-07,
        BLOCK_SIZE
    )
    
    # Step 3: Apply LayerNorm
    output_grid = (batch_size, seq_len, hidden_size)
    fused_layernorm_kernel[output_grid](
        in_2,
        in_3,
        mean_buffer,
        variance_buffer,
        in_0,
        in_1,
        output,
        batch_size,
        seq_len,
        hidden_size,
        1e-07,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_layernorm_forward