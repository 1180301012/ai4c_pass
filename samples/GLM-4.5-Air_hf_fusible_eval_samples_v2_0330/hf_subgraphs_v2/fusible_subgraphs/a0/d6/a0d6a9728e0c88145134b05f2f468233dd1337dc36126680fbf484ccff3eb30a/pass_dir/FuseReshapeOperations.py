import torch
import triton
import triton.language as tl

def pattern(bmm_1, view_shape, transpose_dims, reshape_shape):
    """
    Match the sequence of reshape operations at the end:
    view + transpose + reshape
    
    Original pattern:
    tmp_4 = bmm_1.view(1, seq_len, 1, head_dim)
    tmp_5 = tmp_4.transpose(1, 2)  
    tmp_6 = tmp_5.reshape(1, 1, -1)
    """
    # View operation
    tmp_4 = bmm_1.view(view_shape)
    
    # Transpose operation
    tmp_5 = tmp_4.transpose(transpose_dims[0], transpose_dims[1])
    
    # Reshape operation
    tmp_6 = tmp_5.reshape(reshape_shape)
    
    return tmp_6

def replacement_args(bmm_1, view_shape, transpose_dims, reshape_shape):
    """Extract arguments from matched nodes"""
    return (bmm_1, view_shape, transpose_dims, reshape_shape)

@triton.jit
def fused_reshape_kernel(
    input_ptr, output_ptr,
    batch_size, seq_len, head_dim,
    output_total_size,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel that combines view, transpose, and reshape operations
    Performs the full transformation: [batch, seq_len, head_dim] -> [1, 1, batch*seq_len*head_dim]
    """
    # Each thread handles one element
    pid = tl.program_id(0)
    
    # Calculate input and output offsets
    input_offset = pid
    output_offset = pid
    
    # Calculate indices in input tensor: [batch, seq_len, head_dim]
    total_elements = batch_size * seq_len * head_dim
    if pid >= total_elements:
        return
    
    # Calculate 3D indices from linear offset
    batch_idx = pid // (seq_len * head_dim)
    remainder = pid % (seq_len * head_dim)
    seq_idx = remainder // head_dim
    head_idx = remainder % head_dim
    
    # Apply the transformation steps:
    # 1. view to [1, seq_len, 1, head_dim] 
    # 2. transpose to [1, 1, seq_len, head_dim]
    # 3. reshape to [1, 1, -1]
    
    # For the final [1, 1, batch*seq_len*head_dim] output,
    # each element from input position (batch, seq, head) goes to output position (batch*seq_len*head)
    
    # Handle the case where input has batch_size dimension
    if batch_size > 1:
        # For multiple batches, we need to process all elements
        batch_offset = batch_idx * head_dim * seq_len
        element_offset = seq_idx * head_dim + head_idx
        final_offset = batch_offset + element_offset
    else:
        # Single batch case: direct mapping
        final_offset = pid
    
    # Load input element and store to output
    if pid < total_elements:
        input_val = tl.load(input_ptr + input_offset)
        if final_offset < output_total_size:
            tl.store(output_ptr + final_offset, input_val)

@torch.fx.wrap
def fused_reshape_operations(input_tensor, view_shape, transpose_dims, reshape_shape):
    """
    Wrapper function that fuses view, transpose, and reshape operations
    """
    batch_size, seq_len, head_dim = input_tensor.shape
    
    # Calculate the total number of elements
    input_elements = input_tensor.numel()
    output_elements = reshape_shape[2]
    
    # Determine optimal block size
    BLOCK_SIZE = 1024
    num_programs = (input_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output_tensor = torch.empty(reshape_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    fused_reshape_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        batch_size=batch_size,
        seq_len=seq_len,
        head_dim=head_dim,
        output_total_size=output_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output_tensor

def replacement_func():
    """Return the fused reshape operations function"""
    return fused_reshape_operations