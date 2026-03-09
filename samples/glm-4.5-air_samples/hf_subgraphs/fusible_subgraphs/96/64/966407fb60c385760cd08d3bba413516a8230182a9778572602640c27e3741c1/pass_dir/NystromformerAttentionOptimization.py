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
    tmp_5 = tmp_4.view(-1, 32)  # Dynamic first dimension
    tmp_4 = None
    return (tmp_5,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def nystromformer_attention_kernel(
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
    
    # Compute ranges
    start_m = m * BLOCK_SIZE_M
    start_n = n * BLOCK_SIZE_N
    
    # Create accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    # Iterate over K dimension
    for k in range(0, seq_len, BLOCK_SIZE_K):
        end_k = min(k + BLOCK_SIZE_K, seq_len)
        
        # Load weights [hidden_size, n_channels * kernel_size]
        weight_offset = (m * seq_len + k) * n_channels
        # The weight has shape [batch_size, 1, 65, 1] -> [ batch_size, 65, 1]
        # Reshape for easier access: [batch_size * 65, 1]
        weights = tl.load(weight_ptr + weight_offset, mask=(k < seq_len), other=0.0)
        
        # Load context values [batch_size, seq_len, hidden_size, n_channels]
        # Reshape to [batch_size * seq_len, hidden_size, n_channels]
        context_offset = ((m * seq_len + k) * hidden_size + 0) * n_channels
        context_vals = tl.load(context_ptr + context_offset, mask=(k < seq_len), other=0.0)
        
        # Load value values [batch_size, seq_len, hidden_size, n_channels]
        value_offset = ((m * seq_len + k) * hidden_size + 0) * n_channels
        value_vals = tl.load(value_ptr + value_offset, mask=(k < seq_len), other=0.0)
        
        # Convolution-like operation: point-wise multiply and accumulate
        # This simulates the grouped conv2d operation
        conv_result = weights * value_vals
        
        # Add to context
        add_result = context_vals + conv_result
        
        # Store in accumulator (partial result)
        if start_m + BLOCK_SIZE_M <= batch_size and start_n + BLOCK_SIZE_N <= hidden_size:
            accumulator += add_result[:BLOCK_SIZE_M, :BLOCK_SIZE_K]
    
    # Apply permute operation (0,2,1,3) -> reshape to batch_size, seq_len, n_channels, hidden_size
    # Then transpose to batch_size, n_channels, seq_len, hidden_size
    # Finally reshape to batch_size, seq_len * hidden_size, n_channels
    
    # Apply the final view operation: reshape to [batch_size, seq_len * n_channels, hidden_size]
    if start_m + BLOCK_SIZE_M <= batch_size:
        # Reshape accumulator to match expected output
        result = accumulator.reshape(BLOCK_SIZE_M, -1)
        # Store result
        out_offset = start_m * (seq_len * n_channels * hidden_size) + start_n * hidden_size
        tl.store(out_ptr + out_offset, result, mask=(m < batch_size))

@torch.fx.wrap
def nystromformer_attention_optimized(in_0, in_1, in_2):
    # Get input shapes
    batch_size = in_0.shape[0]
    weight_channels = in_0.shape[2]  # 65
    context_batch, context_seq, context_hidden, context_channels = in_1.shape
    value_batch, value_seq, value_hidden, value_channels = in_2.shape
    
    assert batch_size == context_batch == value_batch
    assert context_seq == value_seq  
    assert context_hidden == value_hidden
    assert context_channels == value_channels
    
    # Final output shape after view operation: [batch_size, seq_len * context_channels, context_hidden]
    output_shape = (batch_size, context_seq * context_channels, context_hidden)
    output = torch.empty(output_shape, dtype=torch.float32, device=in_0.device)
    
    # Set up Triton kernel configuration
    BLOCK_SIZE_M = 4  # Batch dimension
    BLOCK_SIZE_N = 128  # Hidden dimension  
    BLOCK_SIZE_K = 32  # Sequence dimension
    
    # Calculate grid size
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (context_hidden + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_size = (grid_m, grid_n)
    
    nystromformer_attention_kernel[grid_size](
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
    return nystromformer_attention_optimized