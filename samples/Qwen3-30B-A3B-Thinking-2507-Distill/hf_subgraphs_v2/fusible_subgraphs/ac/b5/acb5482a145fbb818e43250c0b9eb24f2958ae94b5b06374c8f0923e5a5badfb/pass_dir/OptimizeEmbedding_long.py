import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: embedding lookup only (single return value)
# ---------------------------------------------------------------------------
def pattern(in_1, in_2):
    return torch.nn.functional.embedding(in_1, in_2, None, None, 2.0, False, False)


def replacement_args(in_1, in_2):
    return (in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel – 2-D grid: axis-0 = tile, axis-1 = index (fast)
#
# BLOCK_SIZE=2048 covers embed_dim=1536 in one tile (512 masked).
# 2-D grid avoids integer division inside the kernel.
# No n_tiles arg needed (n_tiles=1 always for embed_dim=1536, BLOCK_SIZE=2048).
# ---------------------------------------------------------------------------
@triton.jit
def _embedding_kernel(
    indices_ptr,        # [n_indices] int64 – flattened token indices
    weight_ptr,         # [vocab_size, embed_dim] bfloat16, row-major
    out_ptr,            # [n_indices, embed_dim] bfloat16, row-major
    embed_dim,          # runtime: 1536
    BLOCK_SIZE: tl.constexpr,   # 2048
):
    tile  = tl.program_id(0)   # 0 (only 1 tile when BLOCK_SIZE >= embed_dim)
    idx   = tl.program_id(1)   # which token index (fast/contiguous)

    token   = tl.load(indices_ptr + idx)

    offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < embed_dim

    vals = tl.load(weight_ptr + token * embed_dim + offsets, mask=mask)
    tl.store(out_ptr + idx * embed_dim + offsets, vals, mask=mask)


# ---------------------------------------------------------------------------
# Python-level wrapper – @torch.fx.wrap marks it as a leaf (not traced by FX)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _embedding_replacement(in_1, in_2):
    # in_1 : input_ids       [B, S] int64
    # in_2 : embedding weight [vocab, embed_dim] bfloat16

    embed_dim   = in_2.shape[1]              # 1536
    n_indices   = in_1.shape[0] * in_1.shape[1]

    BLOCK_SIZE  = 2048
    n_tiles     = (embed_dim + BLOCK_SIZE - 1) // BLOCK_SIZE  # = 1

    embedding_out = torch.empty((*in_1.shape, embed_dim),
                                dtype=in_2.dtype, device=in_2.device)

    # 2-D grid (1, n_indices) — tile axis first, avoids integer division
    _embedding_kernel[(n_tiles, n_indices)](
        in_1, in_2, embedding_out,
        embed_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return embedding_out


def replacement_func():
    return _embedding_replacement