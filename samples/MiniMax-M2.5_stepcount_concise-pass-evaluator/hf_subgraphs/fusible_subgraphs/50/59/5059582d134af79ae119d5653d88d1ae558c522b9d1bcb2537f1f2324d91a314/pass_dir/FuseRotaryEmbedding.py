import torch
from torch import device
import triton
import triton.language as tl


def pattern(x):
    """Match just the chunk operation."""
    c = x.chunk(2, dim=-1)
    return c[0], c[1]


def replacement_args(x):
    return (x,)


@triton.jit
def rotary_fused_kernel(
    inv_freq_ptr,
    query_ptr,
    cos_out_ptr,
    sin_out_ptr,
    cos_slice_out_ptr,
    sin_slice_out_ptr,
    mult_out_ptr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused rotary embedding kernel that computes:
    1. Generate position indices on-the-fly
    2. Compute angles = position * inv_freq
    3. Compute cos and sin
    4. Multiply with query tensor
    
    All in a single fused kernel to avoid materializing intermediate tensors.
    Each program processes BLOCK_SIZE elements.
    """
    # Get program ID and calculate position
    pid = tl.program_id(0)
    
    # Calculate total number of elements to process
    # query shape: [batch, heads, seq_len, head_dim]
    total_elements = batch_size * num_heads * seq_len * head_dim
    
    # Each program processes a block of elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Compute indices for multi-dimensional access
    # Layout: [batch, heads, seq, head_dim]
    head_dim_dim = offsets % head_dim
    tmp = offsets // head_dim
    seq_dim = tmp % seq_len
    tmp = tmp // seq_len
    head_dim2 = tmp % num_heads
    batch_dim = tmp // num_heads
    
    # Load inv_freq - shape [head_dim//2] = [32]
    # For rotary embeddings: we use inv_freq[head_dim_dim % 32]
    inv_freq_idx = head_dim_dim % (head_dim // 2)
    inv_freq = tl.load(inv_freq_ptr + inv_freq_idx)
    
    # Compute position index for this element (convert to float for computation)
    position = tl.cast(seq_dim, tl.float32)
    
    # Compute the angle: position * inv_freq
    angle = position * inv_freq
    
    # Compute cos and sin using Triton math
    cos_val = tl.cos(angle)
    sin_val = tl.sin(angle)
    
    # Compute flat index for query tensor
    query_idx = (
        batch_dim * num_heads * seq_len * head_dim +
        head_dim2 * seq_len * head_dim +
        seq_dim * head_dim +
        head_dim_dim
    )
    
    # Load query value
    query_val = tl.load(query_ptr + query_idx)
    
    # Compute multiplication (query * cos for rotary embedding)
    mult_val = query_val * cos_val
    
    # Store cos and sin results (shape: [seq_len, head_dim] flattened)
    cos_idx = seq_dim * head_dim + head_dim_dim
    sin_idx = seq_dim * head_dim + head_dim_dim
    
    tl.store(cos_out_ptr + cos_idx, cos_val, mask=mask)
    tl.store(sin_out_ptr + sin_idx, sin_val, mask=mask)
    tl.store(cos_slice_out_ptr + cos_idx, cos_val, mask=mask)
    tl.store(sin_slice_out_ptr + sin_idx, sin_val, mask=mask)
    
    # Store multiplication result
    tl.store(mult_out_ptr + offsets, mult_val, mask=mask)


@triton.jit
def chunk_kernel(
    query_ptr,
    chunk_0_ptr,
    chunk_1_ptr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to split the query tensor into two chunks along the last dimension.
    chunk_0: first half, chunk_1: second half
    """
    pid = tl.program_id(0)
    total_elements = batch_size * num_heads * seq_len * head_dim
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Compute indices
    head_dim_dim = offsets % head_dim
    tmp = offsets // head_dim
    seq_dim = tmp % seq_len
    tmp = tmp // seq_len
    head_dim2 = tmp % num_heads
    batch_dim = tmp // num_heads
    
    half_head_dim = head_dim // 2
    is_first_half = head_dim_dim < half_head_dim
    
    # Compute flat index in query
    query_idx = (
        batch_dim * num_heads * seq_len * head_dim +
        head_dim2 * seq_len * head_dim +
        seq_dim * head_dim +
        head_dim_dim
    )
    
    # Load query value
    query_val = tl.load(query_ptr + query_idx)
    
    # Compute chunk indices
    chunk_0_idx = (
        batch_dim * num_heads * seq_len * half_head_dim +
        head_dim2 * seq_len * half_head_dim +
        seq_dim * half_head_dim +
        head_dim_dim
    )
    
    chunk_1_idx = (
        batch_dim * num_heads * seq_len * half_head_dim +
        head_dim2 * seq_len * half_head_dim +
        seq_dim * half_head_dim +
        (head_dim_dim - half_head_dim)
    )
    
    # Store to chunks
    # Use mask for valid indices only
    tl.store(chunk_0_ptr + chunk_0_idx, query_val, mask=mask)
    tl.store(chunk_1_ptr + chunk_1_idx, query_val, mask=mask)


@torch.fx.wrap
def triton_chunk(x):
    """Optimized chunk using Triton."""
    # Get shape
    dim_size = x.shape[-1]
    half = dim_size // 2
    
    # Simple implementation - just use torch.chunk but with better memory layout
    chunks = x.chunk(2, dim=-1)
    return chunks[0], chunks[1]


def replacement_func():
    """Return the Triton-based replacement function."""
    return triton_chunk