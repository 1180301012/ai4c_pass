import torch
import triton
import triton.language as tl
from torch import device

# Pattern matching function - mirrors the model.py computation exactly
def pattern(in_0, in_1, in_2, in_3):
    conv3d = torch.conv3d(in_3, in_1, in_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    tmp_4 = conv3d.flatten(2)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = in_2.detach()
    tmp_7 = tmp_6.type_as(tmp_5)
    tmp_8 = tmp_7.to(device=device(type='cuda', index=0), copy=True)
    tmp_9 = tmp_5 + tmp_8
    return (tmp_9,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Fused Triton kernel: reads conv3d output in its native layout [1, C, seq_len]
# and adds position embeddings [1, seq_len, C], producing output [1, seq_len, C]
# This avoids intermediate contiguous copies from transpose and type conversion
@triton.jit
def fused_flatten_transpose_add_kernel(
    conv3d_ptr,       # conv3d output: shape [1, C, seq_len], contiguous
    pos_ptr,          # position embeddings: shape [1, seq_len, C], contiguous on GPU
    out_ptr,          # output: shape [1, seq_len, C], contiguous
    hidden_dim,       # C = 768
    seq_len,          # flattened spatial size = 1568
    total_elements,   # seq_len * hidden_dim
    BLOCK_M: tl.constexpr,  # tile size along seq_len dimension
    BLOCK_N: tl.constexpr,  # tile size along hidden_dim dimension
):
    # 2D grid: each program handles a tile of (seq_len, hidden_dim)
    pid_m = tl.program_id(0)  # tile index along seq_len
    pid_n = tl.program_id(1)  # tile index along hidden_dim

    # Compute tile starting positions
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Create offset arrays within the tile
    m_offsets = m_start + tl.arange(0, BLOCK_M)  # seq_len positions
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # hidden_dim positions

    # Masks for boundary handling
    m_mask = m_offsets < seq_len
    n_mask = n_offsets < hidden_dim

    # Output tensor: [1, seq_len, C] - row-major, contiguous
    # Element at (m, n) is at offset m * hidden_dim + n
    out_offsets = m_offsets * hidden_dim + n_offsets
    out_mask = m_mask & n_mask  # 2D mask broadcast

    # Conv3d output: [1, C, seq_len] - row-major, contiguous
    # Element at (c, s) = (n, m) is at offset n * seq_len + m
    # We read the conv3d value that corresponds to output position (m, n)
    conv3d_offsets = n_offsets * seq_len + m_offsets

    # Position embeddings: [1, seq_len, C] - row-major, contiguous
    # Element at (s, c) = (m, n) is at offset m * hidden_dim + n
    pos_offsets = m_offsets * hidden_dim + n_offsets

    # Load values with masks
    conv3d_val = tl.load(conv3d_ptr + conv3d_offsets, mask=out_mask, other=0.0)
    pos_val = tl.load(pos_ptr + pos_offsets, mask=out_mask, other=0.0)

    # Add and store
    result = conv3d_val + pos_val
    tl.store(out_ptr + out_offsets, result, mask=out_mask)


@torch.fx.wrap
def fused_flatten_transpose_add(in_0, in_1, in_2, in_3):
    """
    Fused implementation of:
    conv3d -> flatten(2) -> transpose(1,2) -> type_as -> to(cuda) -> add
    
    Key optimization: Instead of computing intermediate contiguous copies
    for the transposed tensor, we read the conv3d output in its native
    [1, C, seq_len] layout and compute the addition with position embeddings
    using index mapping, writing directly to [1, seq_len, C] output.
    """
    # Step 1: Run conv3d using cuDNN (optimal for this compute-heavy op)
    conv3d_out = torch.conv3d(in_3, in_1, in_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)

    # Determine output shape
    # conv3d_out shape: [1, C, D', H', W']
    C = conv3d_out.shape[1]
    seq_len = conv3d_out.shape[2] * conv3d_out.shape[3] * conv3d_out.shape[4]

    # Prepare position embeddings: detach, type_as, and move to cuda
    # pos_embeds original shape: [1, seq_len, C] on CPU
    pos_embeds = in_2.detach().to(device='cuda', dtype=conv3d_out.dtype)

    # Allocate output tensor: [1, seq_len, C]
    out = torch.empty((1, seq_len, C), dtype=conv3d_out.dtype, device='cuda')

    # Ensure conv3d_out and pos_embeds are contiguous
    conv3d_out = conv3d_out.reshape(1, C, seq_len)  # This is a view (flatten(2))
    if not conv3d_out.is_contiguous():
        conv3d_out = conv3d_out.contiguous()

    if not pos_embeds.is_contiguous():
        pos_embeds = pos_embeds.contiguous()

    # Launch Triton kernel with 2D grid
    BLOCK_M = 32
    BLOCK_N = 64

    grid_m = (seq_len + BLOCK_M - 1) // BLOCK_M
    grid_n = (C + BLOCK_N - 1) // BLOCK_N

    fused_flatten_transpose_add_kernel[(grid_m, grid_n)](
        conv3d_ptr=conv3d_out,
        pos_ptr=pos_embeds,
        out_ptr=out,
        hidden_dim=C,
        seq_len=seq_len,
        total_elements=seq_len * C,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return (out,)


def replacement_func():
    return fused_flatten_transpose_add