import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1):
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    tmp_3 = tmp_2[(slice(None, None, None), slice(1, None, None))]
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
    tmp_5 = tmp_2[(slice(None, None, None), slice(None, -1, None))]
    tmp_6 = torch.nn.functional.pad(tmp_5, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
    tmp_7 = torch.cat([tmp_4, tmp_2, tmp_6], dim=2)
    return (tmp_7,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_embedding_shift_kernel(
    token_ids_ptr,       # [batch, seq_len] int64
    embed_table_ptr,     # [vocab_size, embed_dim] float
    output_ptr,          # [batch, seq_len, 3*embed_dim] float
    batch_size,
    seq_len,
    embed_dim,
    vocab_size,
    stride_token_b,      # stride for batch dim in token_ids
    stride_token_s,      # stride for seq dim in token_ids
    stride_embed_v,      # stride for vocab dim in embed table
    stride_embed_d,      # stride for embed dim in embed table
    stride_out_b,
    stride_out_s,
    stride_out_k,
    BLOCK_B: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Each program handles a tile of (batch, seq_len) positions and a range of embed_dim
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d = tl.program_id(2)

    # Compute the actual indices
    b_start = pid_b * BLOCK_B
    s_start = pid_s * BLOCK_S
    d_start = pid_d * BLOCK_D

    b_offsets = b_start + tl.arange(0, BLOCK_B)
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    d_offsets = d_start + tl.arange(0, BLOCK_D)

    # Mask for bounds
    b_mask = b_offsets < batch_size
    s_mask = s_offsets < seq_len
    d_mask = d_offsets < embed_dim

    # Combined masks
    bs_mask = b_mask & s_mask  # [BLOCK_B, BLOCK_S]

    # Load token IDs: [BLOCK_B, BLOCK_S]
    token_offsets = b_offsets[:, None] * stride_token_b + s_offsets[None, :] * stride_token_s
    token_ids = tl.load(token_ids_ptr + token_offsets, mask=bs_mask, other=0)

    # Compute output offset base: [BLOCK_B, BLOCK_S, BLOCK_D]
    # Output layout: out[b, s, k] where k = segment*embed_dim + d
    # segment 0: right-shifted (token at s+1), segment 1: center (token at s), segment 2: left-shifted (token at s-1)

    # For each segment, we need to look up the appropriate token
    # segment 0 (right neighbor): token at s+1, padded with 0 at s = seq_len-1
    # segment 1 (center): token at s
    # segment 2 (left neighbor): token at s-1, padded with 0 at s = 0

    for segment in range(3):
        if segment == 0:
            # Right neighbor: s+1
            s_neighbor = s_offsets + 1
            neighbor_mask = s_mask & (s_offsets < seq_len - 1)
        elif segment == 1:
            # Center: s
            s_neighbor = s_offsets
            neighbor_mask = s_mask
        else:
            # Left neighbor: s-1
            s_neighbor = s_offsets - 1
            neighbor_mask = s_mask & (s_offsets > 0)

        # For positions where neighbor is out of bounds, token_id should be -1 (pad with 0)
        # Clamp s_neighbor to valid range for token lookup (we'll mask the result anyway)
        # Load neighbor token ids
        neighbor_token_offsets = b_offsets[:, None] * stride_token_b + s_neighbor[None, :] * stride_token_s
        neighbor_token_ids = tl.load(token_ids_ptr + neighbor_token_offsets, mask=b_mask & (s_neighbor >= 0) & (s_neighbor < seq_len), other=0)

        # Lookup embeddings: [BLOCK_B, BLOCK_S, BLOCK_D]
        embed_offsets = neighbor_token_ids[:, None, None] * stride_embed_v + d_offsets[None, None, :] * stride_embed_d
        embed_values = tl.load(embed_table_ptr + embed_offsets, mask=bs_mask[:, None] & d_mask[None, :] & (neighbor_token_ids[:, None] >= 0) & (neighbor_token_ids[:, None] < vocab_size), other=0.0)

        # Apply padding mask: for boundary positions, the neighbor embedding should be 0
        # For segment 0: when s == seq_len-1, output should be 0 (already handled by neighbor_mask)
        # For segment 2: when s == 0, output should be 0 (already handled by neighbor_mask)
        # But we loaded with other=0, so we need to explicitly zero out the invalid positions
        if segment == 0:
            # Zero out the last position (s == seq_len - 1)
            valid = bs_mask & (s_offsets < seq_len - 1)
        elif segment == 1:
            valid = bs_mask
        else:
            # Zero out the first position (s == 0)
            valid = bs_mask & (s_offsets > 0)

        embed_values = embed_values * (valid[:, None] & d_mask[None, :]).to(embed_values.dtype)

        # Store to output
        k_offsets = segment * embed_dim + d_offsets
        out_offsets = b_offsets[:, None, None] * stride_out_b + s_offsets[None, :, None] * stride_out_s + k_offsets[None, None, :] * stride_out_k
        tl.store(output_ptr + out_offsets, embed_values, mask=bs_mask[:, None] & d_mask[None, :])

@torch.fx.wrap
def fused_embedding_shift(token_ids, embed_table):
    batch_size, seq_len = token_ids.shape
    vocab_size, embed_dim = embed_table.shape
    out_dim = 3 * embed_dim

    output = torch.empty((batch_size, seq_len, out_dim), device=embed_table.device, dtype=embed_table.dtype)

    # Choose block sizes based on embed_dim
    BLOCK_D = min(embed_dim, 128)
    BLOCK_S = min(seq_len, 64)
    BLOCK_B = min(batch_size, 8)

    grid_b = (batch_size + BLOCK_B - 1) // BLOCK_B
    grid_s = (seq_len + BLOCK_S - 1) // BLOCK_S
    grid_d = (embed_dim + BLOCK_D - 1) // BLOCK_D

    fused_embedding_shift_kernel[(grid_b, grid_s, grid_d)](
        token_ids_ptr=token_ids,
        embed_table_ptr=embed_table,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        stride_token_b=token_ids.stride(0),
        stride_token_s=token_ids.stride(1),
        stride_embed_v=embed_table.stride(0),
        stride_embed_d=embed_table.stride(1),
        stride_out_b=output.stride(0),
        stride_out_s=output.stride(1),
        stride_out_k=output.stride(2),
        BLOCK_B=BLOCK_B,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
    )

    return output

def replacement_func():
    return fused_embedding_shift