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
    tmp_5 = tmp_4.view(4, 512, 32)
    tmp_4 = None
    return (tmp_5,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def graph5_attention_kernel(
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
    
    # Create accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over sequence dimension
    for seq in range(0, seq_len, 1):
        # Load weight [batch_size, 1, 65, 1] -> flatten for easier access
        weight_offset = (m * seq_len + seq) * n_channels
        weights = tl.load(weight_ptr + weight_offset, mask=True, other=0.0)
        
        # Load context [batch_size, seq_len, hidden_size, n_channels]
        context_offset = ((m * seq_len + seq) * hidden_size + start_n) * n_channels + start_k
        context_vals = tl.load(context_ptr + context_offset, mask=(start_n + BLOCK_SIZE_N <= hidden_size), other=0.0)
        
        # Load value [batch_size, seq_len, hidden_size, n_channels]  
        value_offset = ((m * seq_len + seq) * hidden_size + start_n) * n_channels + start_k
        value_vals = tl.load(value_ptr + value_offset, mask=(start_n + BLOCK_SIZE_N <= hidden_size), other=0.0)
        
        # Simulate grouped conv2d operation: multiply and accumulate
        conv_result = weights * value_vals
        
        # Add to context
        add_result = context_vals + conv_result
        
        # Accumulate results
        if start_m + BLOCK_SIZE_M <= batch_size:
            accumulator += add_result
    
    # Apply permute operation (0,2,1,3): [batch_size, seq_len, hidden_size, n_channels] -> batch_size, n_channels, seq_len, hidden_size
    # Then reshape to [batch_size, seq_len * n_channels, hidden_size]
    
    # Store result in final output format: [4, 512, 32]
    if start_m + BLOCK_SIZE_M <= batch_size:
        # Reshape to match target output shape
        result = accumulator.reshape(BLOCK_SIZE_M, seq_len * n_channels, -1)
        
        # Store with proper offset for [4, 512, 32] output
        out_offset = start_m * (seq_len * n_channels * -1) + start_n * -1
        tl.store(out_ptr + out_offset, result, mask=(m < batch_size))

@torch.fx.wrap
def graph5_attention_optimized(in_0, in_1, in_2):
    # Get input shapes specific to Graph 5
    batch_size = in_0.shape[0]  # 4
    weight_channels = in_0.shape[2]  # 65
    
    context_batch, context_seq, context_hidden, context_channels = in_1.shape  # [4, 4, 512, 8]
    value_batch, value_seq, value_hidden, value_channels = in_2.shape  # [4, 4, 512, 8]
    
    assert batch_size == context_batch == value_batch
    assert context_seq == value_seq  
    assert context_hidden == value_hidden
    assert context_channels == value_channels
    
    # Final output shape for Graph 5: [4, 512, 32]
    output_shape = (batch_size, context_hidden, 32)
    output = torch.empty(output_shape, dtype=torch.float32, device=in_0.device)
    
    # Set up Triton kernel configuration tailored for this graph
    BLOCK_SIZE_M = 1  # Process one batch at a time
    BLOCK_SIZE_N = 128  # Hidden dimension chunk
    BLOCK_SIZE_K = 8   # Channels chunk
    
    # Calculate grid size for Graph 5
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (context_hidden + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (context_channels + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    grid_size = (grid_m, grid_n, grid_k)
    
    graph5_attention_kernel[grid_size](
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
    return graph5_attention_optimized