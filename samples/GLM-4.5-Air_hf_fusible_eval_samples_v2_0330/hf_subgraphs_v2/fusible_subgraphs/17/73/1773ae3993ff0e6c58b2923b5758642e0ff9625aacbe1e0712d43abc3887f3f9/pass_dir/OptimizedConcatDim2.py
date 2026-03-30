import torch
import triton
import triton.language as tl

# Pattern matching function - matches the concatenation operation
def pattern(in_2, in_5, in_3):
    """
    Match the concatenation operation from the computation graph
    Concatenates three tensors along dimension 2
    """
    tmp_2 = torch.cat((in_2, in_5, in_3), dim=2)
    return tmp_2

# Argument extraction function
def replacement_args(in_2, in_5, in_3):
    return (in_2, in_5, in_3)

# Optimized Triton kernel for concatenation along dimension 2
@triton.jit
def concat_kernel(
    in1_ptr,      # Pointer to first input tensor (in_2)
    in2_ptr,      # Pointer to second input tensor (in_5)
    in3_ptr,      # Pointer to third input tensor (in_3)
    out_ptr,      # Pointer to output tensor
    batch,        # Batch size
    height,       # Dimension 0 (typically 1 for these embeddings)
    width1,       # Dimension 2 size of first tensor (in_2)
    width2,       # Dimension 2 size of second tensor (in_5)
    width3,       # Dimension 2 size of third tensor (in_3)
    hidden_size,  # Hidden dimension size (dimension 3)
    BLOCK_SIZE: tl.constexpr
):
    # Program IDs
    pid = tl.program_id(0)
    batch_id = pid // (width1 + width2 + width3)
    pos_id = pid % (width1 + width2 + width3)
    
    # Global index calculation
    batch_offset = batch_id * height * (width1 + width2 + width3) * hidden_size
    pos_offset = pos_id * hidden_size
    global_offset = batch_offset + pos_offset
    
    # Determine which input to read from and calculate source offset
    if pos_id < width1:
        # First input (in_2)
        src_offset = batch_id * height * width1 * hidden_size + pos_id * hidden_size
        data = tl.load(in1_ptr + src_offset, mask=src_offset < batch * height * width1 * hidden_size, other=0.0)
    elif pos_id < width1 + width2:
        # Second input (in_5)
        local_pos = pos_id - width1
        src_offset = batch_id * height * width2 * hidden_size + local_pos * hidden_size
        data = tl.load(in2_ptr + src_offset, mask=src_offset < batch * height * width2 * hidden_size, other=0.0)
    else:
        # Third input (in_3)
        local_pos = pos_id - width1 - width2
        src_offset = batch_id * height * width3 * hidden_size + local_pos * hidden_size
        data = tl.load(in3_ptr + src_offset, mask=src_offset < batch * height * width3 * hidden_size, other=0.0)
    
    # Store to output
    tl.store(out_ptr + global_offset, data, mask=global_offset < batch * height * (width1 + width2 + width3) * hidden_size)

@torch.fx.wrap
def optimized_concat_dim2(in_2, in_5, in_3):
    """
    Optimized concatenation along dimension 2 using Triton
    
    Args:
        in_2: Tensor with shape [batch, height, width1, hidden_size]
        in_5: Tensor with shape [batch, height, width2, hidden_size] 
        in_3: Tensor with shape [batch, height, width3, hidden_size]
    
    Returns:
        Concatenated tensor with shape [batch, height, width1+width2+width3, hidden_size]
    """
    # Get input shapes - from weight meta, we know:
    # in_2: [batch, 1, 1, hidden_size]  (cls_pos_embed)
    # in_5: [batch, 1, seq_len, hidden_size] (patch_pos_embed) 
    # in_3: [batch, 1, seq_len, hidden_size] (det_pos_embed)
    batch = in_2.shape[0]
    height = in_2.shape[1]
    width1 = in_2.shape[2]
    width2 = in_5.shape[2]
    width3 = in_3.shape[2]
    hidden_size = in_2.shape[3]
    
    # Output shape
    out_width = width1 + width2 + width3
    total_elements = batch * height * out_width * hidden_size
    
    # Choose block size - multiple of hidden_size for good memory coalescing
    BLOCK_SIZE = hidden_size
    
    # Calculate grid size - one program per batch position
    grid_size = batch * height * out_width
    
    # Create output tensor
    out = torch.empty((batch, height, out_width, hidden_size), dtype=in_2.dtype, device=in_2.device)
    
    # Launch kernel
    concat_kernel[(grid_size,)](
        in1_ptr=in_2,
        in2_ptr=in_5,
        in3_ptr=in_3,
        out_ptr=out,
        batch=batch,
        height=height,
        width1=width1,
        width2=width2,
        width3=width3,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return optimized_concat_dim2