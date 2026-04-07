import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = in_0 / 2.8284271247461903
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1

def replacement_args(in_0):
    # Extract scale from the pattern that was matched
    scale = 2.8284271247461903
    return (in_0, scale)

@triton.jit
def fused_divide_transpose_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_size,
    n_heads,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate total elements and divide among programs
    total_elements = batch_size * seq_len * hidden_size * n_heads
    n_elements_per_program = total_elements // triton.cdiv(total_elements, BLOCK_SIZE)
    pid = tl.program_id(0)
    offset = pid * n_elements_per_program + tl.arange(0, n_elements_per_program)
    mask = offset < total_elements

    # Calculate linearized indices and unflatten to 4D
    # Order: [batch_size, seq_len, hidden_size, n_heads]
    linear_idx = tl.load(input_ptr + offset, mask=mask, other=0.0)
    
    # Convert to 4D indices
    batch_idx = (offset // (seq_len * hidden_size * n_heads)) % batch_size
    seq_idx = (offset // (hidden_size * n_heads)) % seq_len
    hidden_idx = (offset // n_heads) % hidden_size
    head_idx = offset % n_heads
    
    # Apply fused divide and transpose (swap hidden_size and n_heads)
    # Transpose is achieved by swapping hidden_idx and head_idx during storage
    result = linear_idx / scale
    
    # Calculate new index with transposed dimensions (hidden_size <-> n_heads)
    new_linear_idx = batch_idx * (seq_len * n_heads * hidden_size) + \
                    seq_idx * (n_heads * hidden_size) + \
                    head_idx * hidden_size + \
                    hidden_idx
    
    tl.store(output_ptr + new_linear_idx, result, mask=mask)

@torch.fx.wrap
def fused_divide_transpose(in_0, scale):
    # Get input tensor properties
    batch_size, seq_len, hidden_size, n_heads = in_0.shape
    
    # Create output tensor
    output = torch.empty_like(in_0)
    
    # For very small tensors, use regular operations for correctness
    if in_0.numel() < 1024:
        tmp_0 = in_0 / scale
        return tmp_0.transpose(-1, -2)
    
    # Choose block size based on tensor size
    total_elements = in_0.numel()
    if total_elements < 1000000:
        BLOCK_SIZE = 256
    elif total_elements < 10000000:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    num_programs = triton.cdiv(total_elements, BLOCK_SIZE)
    
    # Launch kernel
    fused_divide_transpose_kernel[(num_programs,)](
        input_ptr=in_0,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        n_heads=n_heads,
        scale=scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_divide_transpose