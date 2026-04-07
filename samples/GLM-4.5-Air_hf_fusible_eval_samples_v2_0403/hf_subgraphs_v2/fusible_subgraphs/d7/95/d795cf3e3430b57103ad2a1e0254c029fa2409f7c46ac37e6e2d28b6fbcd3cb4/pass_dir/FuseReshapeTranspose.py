import torch
import triton
import triton.language as tl

# Pattern matching function for input reshape + transpose
def pattern(in_0):
    tmp_5 = in_0.reshape(1, 19, 7, 19, 7, 96)  # For float16/bfloat16, 128 for float32
    tmp_6 = tmp_5.transpose(2, 3)
    return tmp_6

# Triton kernel for optimized reshape + transpose
@triton.jit
def reshape_transpose_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    h1, w1, h2, w2,  # h1=19, w1=7, h2=19, w2=7 for the spatial decomposition
    channels,  # 96 for float16/bfloat16, 128 for float32
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Each program handles one block
    h_idx = tl.program_id(0)
    w_idx = tl.program_id(1)
    c_idx = tl.program_id(2)
    
    # Compute offset ranges for this block
    start_h = h_idx * BLOCK_SIZE_H
    start_w = w_idx * BLOCK_SIZE_W
    start_c = c_idx * BLOCK_SIZE_C
    
    # Create offset arrays
    h_offsets = start_h + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = start_w + tl.arange(0, BLOCK_SIZE_W)
    c_offsets = start_c + tl.arange(0, BLOCK_SIZE_C)
    
    # Create masks for boundaries
    h_mask = h_offsets < h1
    w_mask = w_offsets < w2  # Transposed: w2 comes from original w1 dimension
    c_mask = c_offsets < channels
    
    # Compute output indices (transposed layout)
    # Original layout: [batch, h1, w1, h2, w2, channels]
    # After transpose: [batch, h1, h2, w2, w1, channels]  (transposed dims 2,3)
    output_h = h_offsets
    output_w = w_offsets  # w_idx now corresponds to transposed w2 dimension
    output_c = c_offsets
    
    # Compute input indices for loading
    # For transpose(2,3): we swap w1 and h2 dimensions in the indexing
    input_h = h_offsets
    input_w1 = w_offsets  # This becomes w2 in transposed view
    input_h2 = output_w   # This was originally h2, now becomes w1 after transpose
    input_w2 = output_h   # This was originally w2, now becomes h1 after transpose
    input_c = c_offsets
    
    # Flatten indices for memory access
    input_offset = (input_h[:, None, None, None, None] * h2 * w1 * channels +
                    input_w1[None, :, None, None, None] * h1 * w1 * channels +
                    input_h2[None, None, :, None, None] * w1 * channels +
                    input_w2[None, None, None, :, None] * channels +
                    input_c[None, None, None, None, :])
    
    output_offset = (output_h[:, None, None, None, None] * h2 * w2 * channels +
                     output_w[None, :, None, None, None] * h1 * w2 * channels +
                     output_c[None, None, :, None, None] * h1 * w2)
    
    # Load input with broadcasting over batch dimension
    input_data = tl.load(input_ptr + input_offset, 
                        mask=(h_mask[:, None, None, None, None] & 
                              w_mask[None, :, None, None, None] &
                              c_mask[None, None, None, None, :]), 
                        other=0.0).to(tl.float32)
    
    # Store output
    tl.store(output_ptr + output_offset, input_data,
             mask=(h_mask[:, None, None, None, None] & 
                   w_mask[None, :, None, None, None] &
                   c_mask[None, None, None, None, :]))

# Kernel wrapper
@torch.fx.wrap
def optimized_reshape_transpose(in_0):
    # Get input dimensions
    batch_size, h_total, w_total, channels = in_0.shape
    
    # Decompose spatial dimensions (133 = 19 * 7)
    h1, w1 = 19, 7
    h2, w2 = 19, 7  # 133 = 19*7 = h1*w1*h2*w2
    
    # Create output tensor with transposed layout
    # Output shape: [batch_size, h1, h2, w2, w1, channels]
    output_shape = (batch_size, h1, h2, w2, w1, channels)
    out = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Set up grid dimensions
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    BLOCK_SIZE_C = 32
    
    grid_h = (h1 + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (h2 + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W  # Transposed dimension
    grid_c = (channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Launch kernel for each batch element
    for b in range(batch_size):
        reshape_transpose_kernel[(grid_h, grid_w, grid_c)](
            input_ptr=in_0[b],
            output_ptr=out[b],
            batch_size=batch_size,
            h1=h1, w1=w1, h2=h2, w2=w2,
            channels=channels,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
        )
    
    return out

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Replacement function
def replacement_func():
    return optimized_reshape_transpose