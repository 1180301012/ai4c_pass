import torch
import triton
import triton.language as tl

def pattern(in_6, tmp_5, tmp_4):
    """Pattern matches linear operation:
    tmp_6 = torch.nn.functional.linear(in_6, tmp_5, tmp_4)
    """
    tmp_6 = torch.nn.functional.linear(in_6, tmp_5, tmp_4)
    return tmp_6

def replacement_args(in_6, tmp_5, tmp_4):
    return (in_6, tmp_5, tmp_4)

@triton.jit
def linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    input_features,
    output_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # Each program handles a tile of the output matrix
    pid = tl.program_id(0)
    
    # Calculate global output position
    row_block = pid % ((batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
    col_block = pid // ((batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
    
    # Define ranges for this program
    row_start = row_block * BLOCK_SIZE_M
    col_start = col_block * BLOCK_SIZE_N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over input features
    for k in range(0, input_features, BLOCK_SIZE_N):
        # Compute effective ranges
        k_block = min(BLOCK_SIZE_N, input_features - k)
        
        # Load weight tile (output_features x k_block)
        weight_tile = tl.load(weight_ptr + col_start * input_features + k + 
                             tl.arange(0, BLOCK_SIZE_N)[:, None] * input_features + 
                             tl.arange(0, k_block)[None, :])
        
        # Load input tile (batch_size x k_block) - but we only need one row at a time
        # We'll process each row in the batch separately
        for m in range(0, min(BLOCK_SIZE_M, batch_size - row_start)):
            row_offset = row_start + m
            input_tile = tl.load(input_ptr + row_offset * input_features + k + tl.arange(0, k_block)[None, :])
            
            # Matrix multiplication on loaded tiles
            accumulator[m, :k_block] += input_tile @ weight_tile[:k_block, :]
    
    # Load bias and add to accumulator
    bias_tile = tl.load(bias_ptr + col_start + tl.arange(0, BLOCK_SIZE_N)[None, :])
    accumulator += bias_tile
    
    # Store output tile
    for m in range(0, min(BLOCK_SIZE_M, batch_size - row_start)):
        for n in range(0, min(BLOCK_SIZE_N, output_features - col_start)):
            output_offset = (row_start + m) * output_features + col_start + n
            tl.store(output_ptr + output_offset, accumulator[m, n])

@triton.jit
def linear_kernel_coalesced(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_features,
    output_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # Each program processes a tile of the output matrix
    pid = tl.program_id(0)
    
    # Calculate tile coordinates
    num_m_blocks = (128 + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M  # Max batch size
    num_n_blocks = (1000 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N  # Output features
    
    m_block = pid % num_m_blocks
    n_block = pid // num_m_blocks
    
    # Tile boundaries
    m_start = m_block * BLOCK_SIZE_M
    m_end = min(m_start + BLOCK_SIZE_M, 128)  # Max batch size
    n_start = n_block * BLOCK_SIZE_N
    n_end = min(n_start + BLOCK_SIZE_N, 1000)  # Output features
    
    # Initialize accumulator for this tile
    accumulator = tl.zeros((m_end - m_start, n_end - n_start), dtype=tl.float32)
    
    # Process input features in blocks
    for k in range(0, input_features, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, input_features)
        
        # Load weight tile: [output_features, input_features] -> [tile_height, k_block]
        weight_tile = tl.load(
            weight_ptr + n_start * input_features + k +
            tl.arange(0, n_end - n_start)[:, None] * input_features +
            tl.arange(0, k_end - k)[None, :]
        )
        
        # Load input tile for all batch elements in this tile: [batch, input_features] -> [tile_height, k_block]
        input_tile = tl.load(
            input_ptr + m_start * input_features + k +
            tl.arange(0, m_end - m_start)[:, None] * input_features +
            tl.arange(0, k_end - k)[None, :]
        )
        
        # Matrix multiplication
        accumulator += tl.dot(input_tile, weight_tile)
    
    # Load and add bias
    bias_tile = tl.load(bias_ptr + n_start + tl.arange(0, n_end - n_start)[None, :])
    accumulator += bias_tile
    
    # Store result
    output_tile = accumulator
    for m in range(m_end - m_start):
        for n in range(n_end - n_start):
            output_offset = (m_start + m) * output_features + n_start + n
            tl.store(output_ptr + output_offset, output_tile[m, n])

@torch.fx.wrap
def optimized_linear(input, weight, bias):
    # Get tensor shapes
    batch_size = input.shape[0] if len(input.shape) > 1 else 1
    input_features = input.shape[-1]
    output_features = bias.shape[0]
    
    # Optimize block sizes for our specific tensor shapes
    # For small batches, use larger tiles to improve occupancy
    BLOCK_SIZE_M = 32    # Output rows per tile
    BLOCK_SIZE_K = 64    # Input features per tile
    BLOCK_SIZE_N = 32    # Output columns per tile
    
    # Calculate grid dimensions
    num_m_blocks = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_n_blocks = (output_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_programs = num_m_blocks * num_n_blocks
    
    # Create output tensor
    output = torch.empty((batch_size, output_features), dtype=input.dtype, device=input.device)
    
    # Launch optimized kernel
    linear_kernel_coalesced[(num_programs,)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        input_features=input_features,
        output_features=output_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    return optimized_linear