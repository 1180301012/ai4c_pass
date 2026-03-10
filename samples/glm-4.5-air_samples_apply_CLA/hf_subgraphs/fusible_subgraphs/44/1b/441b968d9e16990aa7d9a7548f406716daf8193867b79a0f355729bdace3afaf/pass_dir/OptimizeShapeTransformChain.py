import torch
import triton
import triton.language as tl

def pattern(x):
    # Pattern matches: reshape → permute → contiguous
    tmp_reshape = x.reshape(1, 16, 16, -1)  # or (32, 16, 16, -1)
    tmp_permute = tmp_reshape.permute(0, 3, 1, 2)
    out = tmp_permute.contiguous()
    return out  # Return only the final result that matches the original

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_shape_transform_kernel(
    x_ptr, out_ptr, 
    n_batch, n_height, n_width, n_hidden,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the output tensor
    pid = tl.program_id(0)
    batch_idx = pid // (n_height * n_width * n_hidden)
    height_idx = (pid % (n_height * n_width * n_hidden)) // (n_width * n_hidden)
    width_idx = (pid % (n_width * n_hidden)) // n_hidden
    hidden_idx = pid % n_hidden
    
    # Convert output coordinates back to input coordinates
    # Input: [batch, seq_len, hidden_size] where seq_len = n_height * n_width
    # Output: [batch, hidden_size, n_height, n_width]
    
    # Calculate input offset
    seq_len = n_height * n_width
    input_offset = batch_idx * seq_len * n_hidden + height_idx * seq_len + width_idx
    input_hidden_offset = input_offset + hidden_idx
    
    # Create output offset  
    output_offset = batch_idx * n_height * n_width * n_hidden + \
                   height_idx * n_width * n_hidden + width_idx * n_hidden + hidden_idx
    
    mask = height_idx < n_height and width_idx < n_width
    
    # Load from input
    input_mask = input_hidden_offset < (n_batch * seq_len * n_hidden)
    x_val = tl.load(x_ptr + input_hidden_offset, mask=input_mask, other=0.0)
    
    # Store directly to output (effectively doing the reshape and permute in one step)
    tl.store(out_ptr + output_offset, x_val, mask=mask)

@torch.fx.wrap
def optimized_shape_transform(x):
    # Get input shape: [batch, seq_len, hidden_size]
    batch_size, seq_len, hidden_size = x.shape
    
    # Calculate spatial dimensions
    spatial_size = seq_len  # This should be 256, giving us 16x16
    height = 16
    width = 16
    
    # Verify assumptions
    assert seq_len == height * width, f"Expected seq_len={height}*{width}={height*width}, got {seq_len}"
    
    # Output shape: [batch, hidden_size, height, width]
    output_shape = (batch_size, hidden_size, height, width)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Set up kernel parameters
    total_elements = batch_size * height * width * hidden_size
    BLOCK_SIZE = 512
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_shape_transform_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_batch=batch_size,
        n_height=height,
        n_width=width,
        n_hidden=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out  # Return only the final result

def replacement_func():
    return optimized_shape_transform