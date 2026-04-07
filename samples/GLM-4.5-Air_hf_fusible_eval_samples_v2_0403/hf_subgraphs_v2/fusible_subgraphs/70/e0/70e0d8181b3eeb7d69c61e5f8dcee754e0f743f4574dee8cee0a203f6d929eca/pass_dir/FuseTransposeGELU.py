import torch
import triton
import triton.language as tl

# Pattern matching for transpose followed by GELU
def transpose_gelu_pattern(a):
    """Match transpose(-2, -1) followed by gelu pattern"""
    t = a.transpose(-2, -1)
    return torch.nn.functional.gelu(t)

def replacement_args(a):
    """Extract arguments needed for the fused kernel"""
    return (a,)

@triton.jit
def fused_transpose_gelu_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols, n_elements,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """Fused kernel that performs transpose and GELU in one operation"""
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block sizes
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE_M * BLOCK_SIZE_N)
    block_id = pid % num_blocks
    
    # Calculate block coordinates
    block_cols = block_id % tl.cdiv(n_cols, BLOCK_SIZE_N)
    block_rows = block_id // tl.cdiv(n_cols, BLOCK_SIZE_N)
    
    thread_cols = block_cols * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row = block_rows * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # Create mask for bounds checking
    mask_cols = thread_cols < n_cols
    mask_rows = row < n_rows
    
    # Load from input in transposed order (n_cols, n_rows) -> (n_rows, n_cols)
    # Original layout: [batch, seq_len, features] = [1, 3999, 520]
    # After transpose: [batch, features, seq_len] = [1, 512, 3999]
    thread_mask = mask_cols & mask_rows
    
    # Load original data and apply transpose during load
    # input_ptr layout: [1, 3999, 512] -> stride: [3999*512, 512, 1]
    # After transpose we want: [1, 512, 3999] -> stride: [512*3999, 3999, 1]
    input_offset = row[:, None] * n_cols + thread_cols[None, :]
    x = tl.load(input_ptr + input_offset, mask=thread_mask, other=0.0)
    
    # Apply GELU activation
    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_cube = x * x * x
    tanh_arg = 0.7978845608028654 * (x + 0.044715 * x_cube)
    gelu = 0.5 * x * (1 + tl.tanh(tanh_arg))
    
    # Store to output with transposed layout
    # output_ptr layout: [1, 512, 3999] -> stride: [512*3999, 3999, 1]
    output_offset = thread_cols[:, None] * n_rows + row[None, :]
    tl.store(output_ptr + output_offset, gelu, mask=thread_mask[:, None])

@torch.fx.wrap
def fused_transpose_gelu(x):
    """Wrapper for fused transpose+GELU kernel"""
    if x.dim() != 3:
        raise ValueError("Expected 3D input tensor")
    
    # Get tensor dimensions [batch, seq_len, features]
    batch_size, seq_len, features = x.shape
    
    # After transpose, we expect [batch, features, seq_len]
    out_features, out_seq_len = features, seq_len
    
    # Total elements in the output tensor
    n_elements = batch_size * out_features * out_seq_len
    
    # Determine optimal block sizes
    if n_elements > 1000000:  # Large tensor
        BLOCK_SIZE_M = 64   # features dimension
        BLOCK_SIZE_N = 64   # seq_len dimension
    else:  # Small tensor
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
    
    # Calculate number of blocks
    total_blocks = (out_features * out_seq_len + BLOCK_SIZE_M * BLOCK_SIZE_N - 1) // (BLOCK_SIZE_M * BLOCK_SIZE_N)
    
    # Create output tensor with transposed shape
    output = torch.empty((batch_size, out_features, out_seq_len), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    fused_transpose_gelu_kernel[(total_blocks,)](
        input_ptr=x,
        output_ptr=output,
        n_rows=out_features,
        n_cols=out_seq_len,
        n_elements=n_elements,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

def replacement_func():
    """Return the fused transpose+GELU function"""
    return fused_transpose_gelu