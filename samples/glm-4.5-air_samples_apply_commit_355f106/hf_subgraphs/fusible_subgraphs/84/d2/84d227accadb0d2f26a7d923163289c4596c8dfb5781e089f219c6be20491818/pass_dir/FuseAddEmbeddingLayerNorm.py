import torch
import triton
import triton.language as tl


# Simple add kernel for fusing the two addition operations
@triton.jit
def fused_add_kernel(
    in_ptr1, in_ptr2, out_ptr,
    batch_size, seq_len, hidden_size,
    stride_batch1, stride_seq1, stride_hidden1,
    stride_batch2, stride_seq2, stride_hidden2,
    stride_batch_out, stride_seq_out, stride_hidden_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Get position
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    hidden_idx = tl.program_id(2) * BLOCK_SIZE
    
    # Offsets for hidden dimension
    offsets = hidden_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    
    # Calculate offsets for input1
    off1 = batch_idx * stride_batch1 + seq_idx * stride_seq1 + offsets * stride_hidden1
    # Calculate offsets for input2
    off2 = batch_idx * stride_batch2 + seq_idx * stride_seq2 + offsets * stride_hidden2
    # Calculate offsets for output
    off_out = batch_idx * stride_batch_out + seq_idx * stride_seq_out + offsets * stride_hidden_out
    
    # Load and store
    val1 = tl.load(in_ptr1 + off1, mask=mask, other=0.0)
    val2 = tl.load(in_ptr2 + off2, mask=mask, other=0.0)
    result = val1 + val2
    tl.store(out_ptr + off_out, result, mask=mask)


def pattern(in_4, in_6):
    """Match the add operation: inputs_embeds + token_type_embeddings"""
    return in_4 + in_6


def replacement_args(in_4, in_6):
    """Extract arguments for the fused kernel"""
    return (in_4, in_6)


@torch.fx.wrap
def fused_add_wrapper(in_4, in_6):
    """Wrapper for the fused add kernel"""
    batch, seq, hidden = in_4.shape
    output = torch.empty_like(in_4)
    
    # Determine block size based on hidden dimension
    if hidden <= 32:
        block_size = 32
    elif hidden <= 128:
        block_size = 128
    elif hidden <= 512:
        block_size = 512
    else:
        block_size = 1024
    
    num_hidden_blocks = (hidden + block_size - 1) // block_size
    
    # Handle stride broadcasting for in_6
    if in_6.shape[0] == 1:
        stride_batch2 = 0
    else:
        stride_batch2 = in_6.stride(0)
    
    fused_add_kernel[(batch, seq, num_hidden_blocks)](
        in_4, in_6, output,
        batch, seq, hidden,
        in_4.stride(0), in_4.stride(1), in_4.stride(2),
        stride_batch2, in_6.stride(1), in_6.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_SIZE=block_size,
    )
    
    return output


def replacement_func():
    """Return the replacement function"""
    return fused_add_wrapper