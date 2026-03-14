import torch
import triton
import triton.language as tl

def pattern(in_3):
    """
    Pattern matching for tensor chunk operation:
    tmp_8 = in_3.chunk(2, dim=-1)
    tmp_9 = tmp_8[0]
    tmp_10 = tmp_8[1]
    """
    tmp_8 = in_3.chunk(2, dim=-1)
    tmp_9 = tmp_8[0]
    tmp_10 = tmp_8[1]
    return tmp_9, tmp_10

@triton.jit
def chunk_split_kernel(
    in_ptr,
    out1_ptr,      # first chunk output
    out2_ptr,      # second chunk output  
    n_sequences,
    n_heads,
    seq_len,
    total_dim,     # original total dimension (64)
    chunk_dim,     # chunk dimension (32)
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID mapping
    pid = tl.program_id(0)
    stride_n = tl.num_programs(0)
    stride_seq = tl.num_programs(1)
    
    # Calculate indices
    seq_idx = pid // stride_n
    head_idx = (pid % stride_n) // stride_seq
    elem_idx = pid % stride_seq
    
    mask = (seq_idx < n_sequences) & (head_idx < n_heads) & (elem_idx < seq_len)
    
    if mask:
        # Load complete data for this position
        in_offset = seq_idx * n_heads * seq_len * total_dim + head_idx * seq_len * total_dim + elem_idx * total_dim
        in_vals = tl.load(in_ptr + in_offset, mask=tl.arange(0, total_dim) < total_dim, other=0.0)
        
        # Split into two chunks
        chunk1 = in_vals[:chunk_dim]
        chunk2 = in_vals[chunk_dim:]  # Equivalent to in_vals[chunk_dim:chunk_dim*2]
        
        # Store first chunk
        out1_offset = seq_idx * n_heads * seq_len * chunk_dim + head_idx * seq_len * chunk_dim + elem_idx * chunk_dim
        tl.store(out1_ptr + out1_offset, chunk1, mask=tl.arange(0, chunk_dim) < chunk_dim)
        
        # Store second chunk
        out2_offset = seq_idx * n_heads * seq_len * chunk_dim + head_idx * seq_len * chunk_dim + elem_idx * chunk_dim
        tl.store(out2_ptr + out2_offset, chunk2, mask=tl.arange(0, chunk_dim) < chunk_dim)

@torch.fx.wrap
def optimized_chunk_split(input_tensor):
    """
    Optimized kernel for tensor chunk operation splitting into 2 chunks along last dimension
    input_tensor: [n_sequences, n_heads, seq_len, 64]
    output: tuple of ([n_sequences, n_heads, seq_len, 32], [n_sequences, n_heads, seq_len, 32])
    """
    # Get tensor shapes
    n_sequences, n_heads, seq_len, total_dim = input_tensor.shape
    chunk_dim = total_dim // 2  # Should be 32
    
    # Create output tensors
    out1 = torch.empty(n_sequences, n_heads, seq_len, chunk_dim, dtype=input_tensor.dtype, device=input_tensor.device)
    out2 = torch.empty(n_sequences, n_heads, seq_len, chunk_dim, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    total_positions = n_sequences * n_heads * seq_len
    BLOCK_SIZE = 256  # Positions per program
    
    grid = lambda meta: (
        (total_positions + BLOCK_SIZE - 1) // BLOCK_SIZE,
        1, 1
    )
    
    chunk_split_kernel[grid](
        in_ptr=input_tensor,
        out1_ptr=out1,
        out2_ptr=out2,
        n_sequences=n_sequences,
        n_heads=n_heads,
        seq_len=seq_len,
        total_dim=total_dim,
        chunk_dim=chunk_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out1, out2

def replacement_args(in_3):
    return (in_3,)

def replacement_func():
    return optimized_chunk_split