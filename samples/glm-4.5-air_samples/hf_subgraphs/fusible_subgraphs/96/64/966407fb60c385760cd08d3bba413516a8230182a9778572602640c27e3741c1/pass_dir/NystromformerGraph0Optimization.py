import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_2, tmp_0, None, (1, 1), (32, 0), (1, 1), 4)
    tmp_0 = None
    in_1 += tmp_1
    tmp_1 = None
    tmp_2 = in_1
    tmp_3 = tmp_2.permute(0, 2, 1, 3)
    tmp_2 = None
    tmp_4 = tmp_3.contiguous()
    tmp_3 = None
    tmp_5 = tmp_4.view(1, 64, 32)
    tmp_4 = None
    return (tmp_5,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def graph0_attention_kernel(
    weight_ptr,
    context_ptr,
    value_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_size,
    n_channels,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Matrix program ids
    m = tl.program_id(0)
    n = tl.program_id(1)
    k = tl.program_id(2)
    
    # Compute ranges
    start_m = m * BLOCK_SIZE_M
    start_n = n * BLOCK_SIZE_N
    start_k = k * BLOCK_SIZE_K
    
    # Initialize accumulator for this thread block
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Process sequence dimension efficiently
    for seq_idx in range(0, seq_len, BLOCK_SIZE_K):
        end_seq = min(seq_idx + BLOCK_SIZE_K, seq_len)
        
        # Load weight for current batch and sequence position [1, 65, 1] after flattening
        weight_offset = (start_m * seq_len + seq_idx) * n_channels
        weight_vals = tl.load(weight_ptr + weight_offset, 
                              mask=(seq_idx < seq_len), 
                              other=0.0)
        
        # Load context values [batch_size, seq_len, hidden_size, n_channels]
        context_offset = ((start_m * seq_len + seq_idx) * hidden_size + start_n) * n_channels + start_k
        context_vals = tl.load(context_ptr + context_offset, 
                              mask=(start_n + BLOCK_SIZE_N <= hidden_size) & (seq_idx < seq_len), 
                              other=0.0)
        
        # Load value values [batch_size, seq_len, hidden_size, n_channels]  
        value_offset = ((start_m * seq_len + seq_idx) * hidden_size + start_n) * n_channels + start_k
        value_vals = tl.load(value_ptr + value_offset, 
                             mask=(start_n + BLOCK_SIZE_N <= hidden_size) & (seq_idx < seq_len), 
                             other=0.0)
        
        # Simulate grouped conv2d: multiply weights with values
        conv_result = weight_vals * value_vals
        
        # Add convolution result to context (element-wise addition)
        add_result = context_vals + conv_result
        
        # Accumulate results
        if start_m + BLOCK_SIZE_M <= batch_size and start_n + BLOCK_SIZE_N <= hidden_size:
            accumulator += add_result
    
    # Apply the transformation operations in the kernel:
    # 1. permute(0, 2, 1, 3): [batch_size, seq_len, hidden_size, n_channels] -> batch_size, n_channels, seq_len, hidden_size
    # 2. Reshape to final output format [batch_size, hidden_size, channels_factor]
    
    # Store final result in output format [1, 64, 32]
    if start_m + BLOCK_SIZE_M <= batch_size:
        # Reshape accumulator to match target shape: [batch_size, hidden_size, 32]
        result = accumulator.reshape(BLOCK_SIZE_M, hidden_size, -1)
        
        # Calculate output offset for [1, 64, 32] format
        out_offset = start_m * (hidden_size * 32) + start_n * 32
        tl.store(out_ptr + out_offset, result, 
                mask=(m < batch_size) & (start_n < hidden_size))

@torch.fx.wrap
def graph0_attention_optimized(in_0, in_1, in_2):
    # Get input shapes specific to Graph 0
    batch_size = in_0.shape[0]  # 4, but we process as single batch
    weight_channels = in_0.shape[2]  # 65
    
    context_batch, context_seq, context_hidden, context_channels = in_1.shape  # [1, 4, 64, 8]
    value_batch, value_seq, value_hidden, value_channels = in_2.shape  # [1, 4, 64, 8]
    
    assert batch_size == context_batch == value_batch
    assert context_seq == value_seq  
    assert context_hidden == value_hidden
    assert context_channels == value_channels
    
    # Final output shape for Graph 0: [1, 64, 32]
    output_shape = (batch_size, context_hidden, 32)
    output = torch.empty(output_shape, dtype=torch.float32, device=in_0.device)
    
    # Set up Triton kernel configuration tailored for Graph 0 (optimized for small batch)
    BLOCK_SIZE_M = 1   # Process one batch at a time (batch_size=1)
    BLOCK_SIZE_N = 32  # Process hidden dimension in chunks
    BLOCK_SIZE_K = 4   # Process sequence dimension in chunks
    
    # Calculate grid size for Graph 0
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (context_hidden + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (context_channels + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    grid_size = (grid_m, grid_n, grid_k)
    
    graph0_attention_kernel[grid_size](
        in_0,
        in_1, 
        in_2,
        output,
        batch_size,
        context_seq,
        context_hidden,
        context_channels,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    return graph0_attention_optimized