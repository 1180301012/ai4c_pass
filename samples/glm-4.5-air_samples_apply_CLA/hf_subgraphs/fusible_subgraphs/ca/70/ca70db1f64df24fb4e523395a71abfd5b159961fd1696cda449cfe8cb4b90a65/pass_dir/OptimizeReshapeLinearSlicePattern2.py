import torch
import triton
import triton.language as tl

# Pattern 2: Reshape + Linear transformation + Slicing
def pattern(in_4, in_3, in_2):
    tmp_9 = in_4.reshape(300, -1, 256)
    tmp_10 = torch.nn.functional.linear(tmp_9, in_3, in_2)
    tmp_11 = tmp_10[..., :256]
    tmp_12 = tmp_10[..., -256:]
    return tmp_11, tmp_12

def replacement_args(in_4, in_3, in_2):
    return (in_4, in_3, in_2)

@triton.jit
def reshape_linear_slice_kernel2(
    input_ptr,
    weight_ptr,
    bias_ptr,
    out_first_ptr,
    out_second_ptr,
    batch_size,
    input_channels,
    hidden_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program handles a batch of items
    batch_start = tl.program_id(0) * BLOCK_SIZE_M
    batch_end = min(batch_start + BLOCK_SIZE_M, batch_size)
    
    # Process each item in the batch
    for batch_idx in range(batch_start, batch_end):
        # Load input data: reshape from [1, 150, 1, 512] to [300, 1, 256]
        # We need to extract the specific batch item data
        input_offset = batch_idx * input_channels  # 256 channels per item
        
        # Load bias
        bias = tl.load(bias_ptr + batch_idx, other=0.0).to(tl.float32)
        
        # Process both output halves in parallel
        for col_offset in [0, 256]:  # First half (0) and second half (256)
            # Allocate accumulator
            accumulator = tl.zeros((1, hidden_dim), dtype=tl.float32)
            
            # Load weight data for this column offset
            for k in range(0, tl.cdiv(input_channels, BLOCK_SIZE_K)):
                k_start = k * BLOCK_SIZE_K
                k_end = min(k_start + BLOCK_SIZE_K, input_channels)
                k_idx = tl.arange(k_start, k_end)
                k_mask = k_idx < input_channels
                
                # Load input features
                input_data = tl.load(input_ptr + input_offset + k_idx, mask=k_mask, other=0.0).to(tl.float32)
                
                # Load weights for this column range
                weight_cols = tl.arange(col_offset, min(col_offset + hidden_dim, 512))
                weight_mask = weight_cols < 512
                weight_data = tl.load(weight_ptr + k_idx * 512 + weight_cols, 
                                    mask=k_mask.outer(weight_mask), other=0.0).to(tl.float32)
                
                # Matrix multiplication
                accumulator += tl.outer(input_data, weight_data)
            
            # Add bias and store result
            output = accumulator[0, :hidden_dim] + bias
            
            # Store to appropriate output pointer
            if col_offset == 0:
                out_ptr = out_first_ptr
            else:
                out_ptr = out_second_ptr
                
            cols = tl.arange(0, hidden_dim)
            mask = batch_idx < batch_size
            tl.store(out_ptr + batch_idx * hidden_dim + cols, output, mask=mask.outer(cols < hidden_dim))

@torch.fx.wrap
def optimized_reshape_linear_slice2(in_4, in_3, in_2):
    batch_size = 300
    input_channels = 256
    hidden_dim = 256
    
    # Output tensors for both halves
    out_first = torch.empty((batch_size, 1, hidden_dim), device=in_4.device, dtype=torch.float32)
    out_second = torch.empty((batch_size, 1, hidden_dim), device=in_4.device, dtype=torch.float32)
    
    # Grid configuration
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = hidden_dim
    BLOCK_SIZE_K = input_channels
    
    num_programs = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    reshape_linear_slice_kernel2[(num_programs,)](
        in_4,
        in_3,
        in_2,
        out_first,
        out_second,
        batch_size,
        input_channels,
        hidden_dim,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    return out_first, out_second

def replacement_func():
    return optimized_reshape_linear_slice2