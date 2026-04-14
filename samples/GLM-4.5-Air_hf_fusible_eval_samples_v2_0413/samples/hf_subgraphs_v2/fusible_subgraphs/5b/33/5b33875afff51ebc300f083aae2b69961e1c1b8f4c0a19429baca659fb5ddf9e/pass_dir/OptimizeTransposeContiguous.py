import torch
import triton
import triton.language as tl

def pattern(attention_output):
    # Transpose and then make contiguous
    transposed = attention_output.permute(0, 2, 1, 3)
    contiguous = transposed.contiguous()
    return contiguous

def replacement_args(attention_output):
    return (attention_output,)

@triton.jit
def optimized_transpose_contiguous_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1, 
    input_stride_2,
    input_stride_3,
    
    output_ptr,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    output_stride_3,
    
    batch_size,
    num_heads,
    seq_in,
    head_dim,
    
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Compute program ids - transpose from [B,H,S_in,D] to [B,S_in,H,D]
    batch_id = tl.program_id(0)
    seq_in_id = tl.program_id(1)
    head_dim_id = tl.program_id(2)
    
    # Within block coordinates
    x_off = tl.program_id(3) * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    y_off = tl.program_id(4) * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    
    # Compute input and output pointers for current batch and position
    input_ptr += batch_id * input_stride_0 + seq_in_id * input_stride_2 + head_dim_id * input_stride_3
    output_ptr += batch_id * output_stride_0 + seq_in_id * output_stride_1 + head_dim_id * output_stride_3
    
    # Load input block: [H, BLOCK_SIZE_X]
    input_block = tl.load(
        input_ptr + y_off[:, None] + x_off[None, :],
        mask=(y_off[:, None] < num_heads) & (x_off[None, :] < BLOCK_SIZE_X),
        other=0.0
    )
    
    # Store output block: [BLOCK_SIZE_Y, BLOCK_SIZE_X]
    output_block = input_block
    tl.store(
        output_ptr + x_off[:, None] + y_off[None, :],
        output_block,
        mask=(x_off[:, None] < BLOCK_SIZE_X) & (y_off[None, :] < BLOCK_SIZE_Y),
    )

@torch.fx.wrap
def optimized_transpose_contiguous(attention_output):
    # Get tensor shapes
    batch_size, num_heads, seq_len, head_dim = attention_output.shape
    
    # Set up kernel configuration
    BLOCK_SIZE_X = 64  # head_dim dimension
    BLOCK_SIZE_Y = 8   # num_heads dimension
    
    # Calculate grid dimensions
    grid_x = (head_dim + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    grid_y = (num_heads + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    
    # Create output tensor
    transposed_output = torch.empty_like(attention_output).permute(0, 2, 1, 3)
    
    # Launch kernel with proper grid
    grid = (batch_size, seq_len, head_dim, grid_x, grid_y)
    optimized_transpose_contiguous_kernel[grid](
        attention_output,
        attention_output.stride(0), attention_output.stride(1), attention_output.stride(2), attention_output.stride(3),
        transposed_output,
        transposed_output.stride(0), transposed_output.stride(1), transposed_output.stride(2), transposed_output.stride(3),
        batch_size, num_heads, seq_len, head_dim,
        BLOCK_SIZE_X, BLOCK_SIZE_Y
    )
    
    return transposed_output

def replacement_func():
    return optimized_transpose_contiguous