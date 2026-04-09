import torch
import triton
import triton.language as tl

def pattern(embeddings, indices, seq_len, hidden_size):
    """
    Pattern to match: arange + unsqueeze + add + view + index_select + view
    
    This matches the sequence:
    tmp_4 = torch.arange(0, 9, dtype=torch.int64, device=device(type='cuda', index=0))
    tmp_5 = tmp_4.unsqueeze(0)
    tmp_5 += 2
    tmp_7 = tmp_5.view(-1)
    tmp_8 = embeddings.index_select(0, tmp_7)  # embeddings is tmp_1
    tmp_9 = tmp_8.view(1, 9, hidden_size)
    
    Returns the final tensor after all operations
    """
    # Create index tensor
    tmp_4 = torch.arange(0, seq_len, dtype=torch.int64, device=embeddings.device)
    tmp_5 = tmp_4.unsqueeze(0)
    tmp_5 += 2
    tmp_7 = tmp_5.view(-1)
    
    # Index select and view
    tmp_8 = embeddings.index_select(0, tmp_7)
    tmp_9 = tmp_8.view(1, seq_len, hidden_size)
    
    return tmp_9

def replacement_args(embeddings, indices, seq_len, hidden_size):
    """Extract arguments needed for replacement"""
    return (embeddings, indices, seq_len, hidden_size)

@triton.jit
def slice_kernel(
    embeddings_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_size: tl.constexpr,
    stride_embeddings: tl.constexpr,
    stride_out: tl.constexpr,
):
    """Triton kernel to directly slice embeddings tensor"""
    pid = tl.program_id(0)
    
    # Process each sequence position
    for i in tl.range(0, seq_len, num_warps=4):
        row_offset = 2 + i  # Start from row 2 and take consecutive rows
        
        # Each thread processes a hidden dimension chunk
        hidden_offset = pid * hidden_size + tl.arange(0, hidden_size)
        mask = hidden_offset < hidden_size
        
        # Load embedding data directly from sliced region
        embeddings_data = tl.load(
            embeddings_ptr + row_offset * stride_embeddings + hidden_offset,
            mask=mask,
            other=0.0
        )
        
        # Store output
        tl.store(
            out_ptr + i * stride_out + hidden_offset,
            embeddings_data,
            mask=mask
        )

@torch.fx.wrap
def direct_slice_embedding(embeddings, seq_len, hidden_size):
    """
    Optimized function that directly slices embeddings tensor
    instead of using expensive index_select
    """
    batch_size = 1
    output_shape = (1, seq_len, hidden_size)
    out = torch.empty(output_shape, dtype=embeddings.dtype, device=embeddings.device)
    
    # Calculate strides
    stride_embeddings = embeddings.stride(0)  # stride for first dimension
    stride_out = out.stride(1)  # stride for sequence dimension
    
    # Launch kernel
    n_hidden_blocks = (hidden_size + 1023) // 1024
    slice_kernel[(n_hidden_blocks,)](
        embeddings_ptr=embeddings,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        stride_embeddings=stride_embeddings,
        stride_out=stride_out,
    )
    
    return out

def replacement_func():
    """Return the optimized replacement function"""
    def optimized_forward(embeddings, indices, seq_len, hidden_size):
        """
        Optimized replacement that uses direct slicing instead of index_select
        """
        return direct_slice_embedding(embeddings, seq_len, hidden_size)
    
    return optimized_forward