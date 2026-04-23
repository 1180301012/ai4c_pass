import torch
import triton
import triton.language as tl


@triton.jit
def fused_slice_pad_cat_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    seq_len,
    embed_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    - tmp_3 = in[:, 1:, :]  (slice from index 1)
    - tmp_4 = pad(tmp_3, [0,0,0,1,0,0])  (pad dim 1 with 1 zero at end)
    - tmp_5 = in[:, :-1, :]  (slice before last)
    - tmp_6 = pad(tmp_5, [0,0,1,0,0,0])  (pad dim 1 with 1 zero at start)
    - output = cat([tmp_4, in, tmp_6], dim=2)
    
    All in a single kernel to avoid memory traffic for intermediates.
    Output shape: [batch, seq_len, embed_dim * 3]
    """
    # Calculate global thread index
    pid = tl.program_id(0)
    
    # Each thread handles a (batch_idx, seq_idx) position
    total_positions = batch_size * seq_len
    position_idx = pid
    
    if position_idx >= total_positions:
        return
    
    batch_idx = position_idx // seq_len
    seq_idx = position_idx % seq_len
    
    # Calculate output offsets for each of the 3 parts
    # Output layout: [batch, seq, embed*3]
    # Part 1 (tmp_4): from in[:, 1:, :] padded with 1 zero at end
    # Part 2 (in): direct copy
    # Part 3 (tmp_6): from in[:, :-1, :] padded with 1 zero at start
    
    embed_dim_3 = embed_dim * 3
    
    # Load input data for this (batch, seq) position
    for emb_idx in range(embed_dim):
        in_offset = batch_idx * seq_len * embed_dim + seq_idx * embed_dim + emb_idx
        
        # Value from original embedding
        val = tl.load(in_ptr + in_offset)
        
        # Calculate source index for tmp_3 slice (in[:, 1:, :])
        # tmp_3[:, i, :] corresponds to in[:, i+1, :]
        # tmp_4 pads with 0 at end, so tmp_4[i] comes from tmp_3[i] for i < seq_len-1
        # For tmp_4, we need: out[0..seq_len-2] = in[1..seq_len-1], out[seq_len-1] = 0
        
        # Part 1: tmp_4 (padded left slice)
        if seq_idx < seq_len - 1:
            # in[:, 1:, :] source: in[:, seq_idx+1, :]
            src_seq_idx = seq_idx + 1
            src_offset = batch_idx * seq_len * embed_dim + src_seq_idx * embed_dim + emb_idx
            part1_val = tl.load(in_ptr + src_offset)
        else:
            # Last position padded with 0
            part1_val = 0.0
        
        # Part 2: direct copy of in
        part2_val = val
        
        # Part 3: tmp_6 (padded right slice)
        if seq_idx > 0:
            # in[:, :-1, :] source: in[:, seq_idx-1, :]
            src_seq_idx = seq_idx - 1
            src_offset = batch_idx * seq_len * embed_dim + src_seq_idx * embed_dim + emb_idx
            part3_val = tl.load(in_ptr + src_offset)
        else:
            # First position padded with 0
            part3_val = 0.0
        
        # Store to output: [part1, part2, part3]
        out_base = batch_idx * seq_len * embed_dim_3 + seq_idx * embed_dim
        tl.store(out_ptr + out_base + emb_idx, part1_val)
        tl.store(out_ptr + out_base + embed_dim + emb_idx, part2_val)
        tl.store(out_ptr + out_base + embed_dim * 2 + emb_idx, part3_val)


@triton.jit
def fused_slice_pad_cat_kernel_autotuned(
    in_ptr,
    out_ptr,
    batch_size,
    seq_len,
    embed_dim,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Autotuned version of the fused slice-pad-cat kernel.
    Uses BLOCK_SIZE for the inner embedding loop.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load the values
    vals = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Store to output
    tl.store(out_ptr + offsets, vals, mask=mask)


@torch.fx.wrap
def fused_slice_pad_cat_wrapper(in_tensor):
    """
    Wrapper for the fused slice-pad-cat operation.
    
    Args:
        in_tensor: Input tensor of shape [batch, seq_len, embed_dim]
        
    Returns:
        Output tensor of shape [batch, seq_len, embed_dim * 3]
    """
    batch_size, seq_len, embed_dim = in_tensor.shape
    output_dim = embed_dim * 3
    
    # Allocate output
    out = torch.empty(
        (batch_size, seq_len, output_dim),
        dtype=in_tensor.dtype,
        device=in_tensor.device
    )
    
    # Calculate grid
    # Each program processes one (batch, seq) position, processing all embed_dim elements
    total_positions = batch_size * seq_len
    
    # Use autotune for block size
    BLOCK_SIZE = 128
    
    num_programs = total_positions
    
    fused_slice_pad_cat_kernel[(num_programs,)](
        in_ptr=in_tensor,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1):
    """
    Match the embedding followed by slice-pad-slice-pad-cat pattern.
    
    Operations:
    - tmp_2 = embedding(in_0, in_1, ...)
    - tmp_3 = tmp_2[:, 1:, :]  (slice from index 1)
    - tmp_4 = pad(tmp_3, [0,0,0,1,0,0])  (pad dim 1 with 1 zero at end)
    - tmp_5 = tmp_2[:, :-1, :]  (slice before last)
    - tmp_6 = pad(tmp_5, [0,0,1,0,0,0])  (pad dim 1 with 1 zero at start)
    - tmp_7 = cat([tmp_4, tmp_2, tmp_6], dim=2)
    
    Return tmp_7 which is the final concatenated output.
    """
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    
    # Slice from index 1 onwards
    tmp_3 = tmp_2[(slice(None, None, None), slice(1, None, None))]
    # Pad with 1 zero at end of dim 1
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
    
    # Slice before last
    tmp_5 = tmp_2[(slice(None, None, None), slice(None, -1, None))]
    # Pad with 1 zero at start of dim 1
    tmp_6 = torch.nn.functional.pad(tmp_5, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
    
    # Concatenate along dim 2
    tmp_7 = torch.cat([tmp_4, tmp_2, tmp_6], dim=2)
    
    return tmp_7


def replacement_args(in_0, in_1):
    """
    Extract arguments needed for replacement.
    We return both the embedding indices and the embedding table since the fused kernel
    needs to read the embedding table.
    """
    return (in_0, in_1)


def replacement_func():
    """
    Returns the replacement function that uses a fused kernel.
    
    The original computation is:
    1. embedding lookup
    2. slice from 1, pad end
    3. slice to -1, pad start
    4. cat all three along dim 2
    
    This fuses steps 2-4 into a single kernel to avoid materializing intermediates.
    """
    def optimized_embedding_slice_pad_cat(in_0, in_1):
        """
        Optimized version that fuses the slice-pad-cat pattern.
        The embedding is done by PyTorch's optimized implementation.
        Only the slice-pad-cat is fused into a custom kernel.
        """
        # Use PyTorch's optimized embedding
        embedded = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
        
        # Use fused kernel for slice-pad-cat
        output = fused_slice_pad_cat_wrapper(embedded)
        
        return output
    
    return optimized_embedding_slice_pad_cat