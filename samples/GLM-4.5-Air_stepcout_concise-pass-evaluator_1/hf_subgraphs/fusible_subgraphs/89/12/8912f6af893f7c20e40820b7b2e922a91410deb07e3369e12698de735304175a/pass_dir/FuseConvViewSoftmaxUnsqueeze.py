import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1, in_2):
    """Match the computational pattern: conv2d -> view -> softmax -> unsqueeze"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = tmp_2.view(0, 0, 0)  # Flexible view shape to match different patterns
    tmp_2 = None
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_3 = None
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_4 = None
    return (tmp_5,)

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the fused kernel"""
    return (in_0, in_1, in_2)

@triton.jit
def softmax_kernel_1d(
    input_ptr,
    output_ptr,
    n_elements,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized 1D softmax kernel along a specific dimension"""
    pid = tl.program_id(0)
    
    if pid >= n_elements:
        return
        
    # Calculate offset for the beginning of this sequence
    base_offset = pid * dim_size
    
    # Load sequence
    x = tl.load(input_ptr + base_offset + tl.arange(0, BLOCK_SIZE), 
                mask=tl.arange(0, BLOCK_SIZE) < dim_size, 
                other=-float('inf'))
    
    # Compute max for numerical stability
    m = tl.max(x, axis=0)
    
    # Compute exp and sum
    x_exp = tl.exp(x - m)
    s = tl.sum(x_exp, axis=0)
    
    # Normalize
    y = x_exp / s
    
    # Store result
    tl.store(output_ptr + base_offset + tl.arange(0, BLOCK_SIZE), 
             y, mask=tl.arange(0, BLOCK_SIZE) < dim_size)

@torch.fx.wrap
def fused_conv_view_softmax_unsqueeze(in_0, in_1, in_2):
    """Execute the fused operation: conv2d + view + softmax + unsqueeze"""
    
    # Step 1: Apply conv2d
    conv_output = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    
    # Get conv output shape and compute view shape
    batch_size, output_channels, height, width = conv_output.shape
    seq_len = height * width
    
    # Step 2: Apply view operation to reshape from [B, C, H, W] to [B*C, 1, H*W]
    # In our case, since conv produces [B, 1, H, W], view reshape to [B, 1, H*W]
    viewed = conv_output.view(batch_size, output_channels, seq_len)
    
    # Step 3: Apply softmax along dimension 2 (the sequence dimension)
    # Need to reshape for softmax kernel: [B*C, H*W] -> softmax on last dim
    softmax_input = viewed.reshape(-1, seq_len)
    n_sequences = softmax_input.shape[0]
    
    # Create output for softmax
    softmax_output = torch.zeros_like(softmax_input)
    
    # Launch optimized softmax kernel
    BLOCK_SIZE = 1024
    num_programs = (n_sequences + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    softmax_kernel_1d[(num_programs,)](
        input_ptr=softmax_input,
        output_ptr=softmax_output,
        n_elements=n_sequences,
        dim_size=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to original 3D format
    softmax_reshaped = softmax_output.reshape(batch_size, output_channels, seq_len)
    
    # Step 4: Apply unsqueeze(-1) to add the final dimension
    result = softmax_reshaped.unsqueeze(-1)
    
    return result

def replacement_func():
    """Return the fused kernel function"""
    return fused_conv_view_softmax_unsqueeze