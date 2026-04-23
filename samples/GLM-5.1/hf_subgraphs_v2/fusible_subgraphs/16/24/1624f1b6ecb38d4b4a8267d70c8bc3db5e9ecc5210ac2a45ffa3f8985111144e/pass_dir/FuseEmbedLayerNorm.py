import torch
import triton
import triton.language as tl

# Match: flatten + transpose + tile + cat + add (proven to work)
# Strategy: use a two-kernel approach for better memory coalescing:
# 1. First transpose conv3d output to contiguous [1, 980, 768] layout
# 2. Then fuse cat + add in a contiguous kernel

def pattern(conv3d_output, cls_token, pos_emb):
    tmp_7 = conv3d_output.flatten(2)
    tmp_8 = tmp_7.transpose(1, 2)
    tmp_9 = cls_token.tile([1, 1, 1])
    tmp_10 = torch.cat((tmp_9, tmp_8), dim=1)
    tmp_11 = tmp_10 + pos_emb
    return tmp_11


def replacement_args(conv3d_output, cls_token, pos_emb):
    return (conv3d_output, cls_token, pos_emb)


# Kernel 1: Transpose conv3d output from [M, N] to [N, M] with good coalescing
@triton.jit
def transpose_kernel(
    input_ptr,
    output_ptr,
    M,  # channels (768)
    N,  # spatial (980)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Read input tile [BLOCK_M, BLOCK_N]
    input_offsets = m_offsets[:, None] * N + n_offsets[None, :]
    mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tile = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)

    # Write transposed tile [BLOCK_N, BLOCK_M]
    output_offsets = n_offsets[:, None] * M + m_offsets[None, :]
    out_mask = (n_offsets[:, None] < N) & (m_offsets[None, :] < M)
    tl.store(output_ptr + output_offsets, tile.trans(), mask=out_mask)


# Kernel 2: Fuse cat(cls_token, transposed) + add(pos_emb) with contiguous reads
@triton.jit
def fused_cat_add_kernel(
    transposed_ptr,    # [1, 980, 768] contiguous
    cls_token_ptr,     # [1, 1, 768]
    pos_emb_ptr,       # [1, 981, 768] contiguous
    out_ptr,           # [1, 981, 768] contiguous
    spatial_size,      # 980
    n_cols,            # 768
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    # Total rows = spatial_size + 1 (cls_token) = 981
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols

    if row_idx == 0:
        # Row 0: cls_token values
        vals = tl.load(cls_token_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    else:
        # Row 1..spatial_size: transposed conv3d values (contiguous!)
        transposed_row_offset = (row_idx - 1) * n_cols
        vals = tl.load(transposed_ptr + transposed_row_offset + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    # Add position embeddings (contiguous)
    pos_row_offset = row_idx * n_cols
    pos_vals = tl.load(pos_emb_ptr + pos_row_offset + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    vals = vals + pos_vals

    out_row_offset = row_idx * n_cols
    tl.store(out_ptr + out_row_offset + col_offsets, vals, mask=col_mask)


@torch.fx.wrap
def fused_embed(conv3d_output, cls_token, pos_emb):
    channels = conv3d_output.shape[1]
    total = conv3d_output.numel()
    spatial_size = total // channels
    n_rows = spatial_size + 1
    n_cols = channels

    # Step 1: Transpose conv3d output to contiguous layout
    transposed = torch.empty((1, spatial_size, n_cols), dtype=conv3d_output.dtype, device=conv3d_output.device)

    BLOCK_M = 32
    BLOCK_N = 32
    grid_m = triton.cdiv(channels, BLOCK_M)
    grid_n = triton.cdiv(spatial_size, BLOCK_N)
    grid = (grid_m, grid_n)

    transpose_kernel[grid](
        conv3d_output, transposed,
        M=channels, N=spatial_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )

    # Step 2: Fuse cat + add
    out = torch.empty((1, n_rows, n_cols), dtype=conv3d_output.dtype, device=conv3d_output.device)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid2 = (n_rows,)

    fused_cat_add_kernel[grid2](
        transposed, cls_token, pos_emb, out,
        spatial_size=spatial_size, n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_embed