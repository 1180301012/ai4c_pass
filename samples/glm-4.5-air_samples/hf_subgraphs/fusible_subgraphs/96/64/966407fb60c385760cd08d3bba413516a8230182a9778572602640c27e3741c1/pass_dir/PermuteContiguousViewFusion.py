import torch
import triton
import triton.language as tl

def pattern(in_1):
    tmp_2 = in_1
    tmp_3 = tmp_2.permute(0, 2, 1, 3)
    tmp_2 = None
    tmp_4 = tmp_3.contiguous()
    tmp_3 = None
    return (tmp_4,)

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def permute_contiguous_view_kernel(
    input_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_size,
    n_channels,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # Matrix program ids
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Compute ranges
    start_m = m * BLOCK_SIZE_M
    start_n = n * BLOCK_SIZE_N
    
    # Create output accumulator
    if start_m < batch_size and start_n < seq_len * n_channels * hidden_size:
        # Calculate effective indices after permute(0,2,1,3) operation
        # Original: [batch_size, seq_len, hidden_size, n_channels]
        # After permute: [batch_size, hidden_size, seq_len, n_channels] 
        # Then reshape to: [batch_size, seq_len * n_channels, hidden_size]
        
        output_idx = start_m * (seq_len * n_channels * hidden_size) + start_n
        
        # If we're in the valid range for output
        seq_chan_idx = start_n // hidden_size  # Combined sequence and channel index
        hidden_idx = start_n % hidden_size       # Hidden dimension
        
        # Calculate original indices for permute(0,2,1,3)
        # [batch, seq, hidden, channels] -> [batch, hidden, seq, channels]
        # Reshaped to: [batch, seq*channels, hidden]
        seq_idx = seq_chan_idx // n_channels    # Original sequence index
        chan_idx = seq_chan_idx % n_channels    # Channel index
        
        # Load original value at [batch, seq, hidden, channels]
        input_offset = ((start_m * seq_len + seq_idx) * hidden_size + hidden_idx) * n_channels + chan_idx
        input_val = tl.load(input_ptr + input_offset, 
                           mask=(start_m < batch_size) & (seq_idx < seq_len) & (hidden_idx < hidden_size), 
                           other=0.0)
        
        # Store result directly in final format
        tl.store(out_ptr + output_idx, input_val, mask=(start_m < batch_size) & (start_n < seq_len * n_channels * hidden_size))

@torch.fx.wrap  
def permute_contiguous_view_optimized(in_1):
    # Get input shape information
    batch_size, seq_len, hidden_size, n_channels = in_1.shape
    # Create output tensor with target shape
    output = torch.empty((batch_size, seq_len, hidden_size, n_channels), dtype=torch.float32, device=in_1.device)
    
    # Set up Triton kernel configuration
    BLOCK_SIZE_M = 1   # Process one batch at a time
    BLOCK_SIZE_N = 128 # Process output dimension in chunks
    
    # Calculate grid size
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (seq_len * n_channels * hidden_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_size = (grid_m, grid_n)
    
    permute_contiguous_view_kernel[grid_size](
        in_1,
        output,
        batch_size,
        seq_len,
        hidden_size,
        n_channels,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    return permute_contiguous_view_optimized