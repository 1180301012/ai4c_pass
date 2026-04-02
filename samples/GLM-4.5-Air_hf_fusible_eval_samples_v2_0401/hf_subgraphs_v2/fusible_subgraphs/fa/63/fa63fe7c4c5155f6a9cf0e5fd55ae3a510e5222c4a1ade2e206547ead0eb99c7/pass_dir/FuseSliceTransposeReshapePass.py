import torch
import triton
import triton.language as tl

@triton.jit
def optimized_transpose_reshape_kernel(
    input_ptr, output_ptr,
    batch_size: tl.constexpr, heads: tl.constexpr,
    feat_dim: tl.constexpr, seq_len: tl.constexpr,
    new_feat_dim: tl.constexpr, height: tl.constexpr, width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Total elements to process
    total_elements = batch_size * heads * feat_dim * seq_len
    elements_per_program = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    start_idx = pid * elements_per_program
    end_idx = min((pid + 1) * elements_per_program, total_elements)
    
    # Process each element
    for idx in range(start_idx, end_idx):
        # Calculate indices
        batch = idx // (heads * feat_dim * seq_len)
        remainder = idx % (heads * feat_dim * seq_len)
        head = remainder // (feat_dim * seq_len)
        remainder = remainder % (feat_dim * seq_len)
        feat = remainder // seq_len
        seq = remainder % seq_len
        
        # Transpose: swap feat and seq dimensions
        # Calculate output position after reshape
        linear_idx = feat * seq_len + seq
        
        # Map to spatial dimensions
        elem_idx = linear_idx
        w = elem_idx % width
        elem_idx //= width
        h = elem_idx % height
        elem_idx //= height
        new_f = elem_idx % new_feat_dim
        
        # Load input value
        input_offset = batch * heads * feat_dim * seq_len + head * feat_dim * seq_len + feat * seq_len + seq
        value = tl.load(input_ptr + input_offset, mask=(feat < feat_dim) & (seq < seq_len), other=0.0)
        
        # Store in output
        output_offset = batch * heads * new_feat_dim * height * width + head * new_feat_dim * height * width + new_f * height * width + h * width + w
        tl.store(output_ptr + output_offset, value)

@torch.fx.wrap
def optimized_transpose_reshape(x, reshape_shape):
    """
    Optimize: transpose(-1, -2) followed by reshape
    """
    batch_size, heads, seq_len, feat_dim = x.shape
    new_feat_dim, height, width = reshape_shape[1], reshape_shape[2], reshape_shape[3]
    
    # Verify reshape is valid
    assert seq_len * feat_dim == new_feat_dim * height * width, "Invalid reshape dimensions"
    
    # Create output tensor
    output = torch.empty(batch_size, heads, new_feat_dim, height, width, 
                        dtype=x.dtype, device=x.device)
    
    # Set kernel configuration
    BLOCK_SIZE = 1024
    
    # Calculate grid size using regular Python math
    total_elements = batch_size * heads * feat_dim * seq_len
    grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_transpose_reshape_kernel[grid](
        x, output,
        batch_size, heads, feat_dim, seq_len,
        new_feat_dim, height, width,
        BLOCK_SIZE
    )
    
    return output

def pattern(x):
    tmp_1 = x.transpose(-1, -2)
    tmp_2 = tmp_1.reshape(1, 256, 8, 8)  # Common dimension for pattern matching
    return tmp_2

def replacement_args(x):
    return (x,)

def replacement_func():
    return optimized_transpose_reshape