import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton embedding lookup kernel
# For each (token_idx, d_block) pair we load one slice of the embedding
# row and store it into the output.  This is memory-bandwidth-bound.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 16,  'BLOCK_D': 512},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_N': 16,  'BLOCK_D': 512},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_N': 32,  'BLOCK_D': 512},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_N': 32,  'BLOCK_D': 512},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_N': 16,  'BLOCK_D': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 16,  'BLOCK_D': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_N': 32,  'BLOCK_D': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 32,  'BLOCK_D': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_N': 64,  'BLOCK_D': 512},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 64,  'BLOCK_D': 512},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_N': 64,  'BLOCK_D': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 64,  'BLOCK_D': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_N': 128, 'BLOCK_D': 512},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_D': 512},  num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 256, 'BLOCK_D': 512},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_D': 512},  num_stages=2, num_warps=8),
    ],
    key=['N', 'D'],
)
@triton.jit
def _embedding_fwd(
    idx_ptr,   # [*seq_shape] int64  – token indices (flattened)
    emb_ptr,   # [vocab_size, D]      – embedding table
    out_ptr,   # [*seq_shape, D]      – output (flattened)
    N,         # total number of tokens (prod of seq_shape)
    D,         # embedding dimension
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Which tokens does this program handle?
    row_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    row_mask = row_idx < N

    # Which slice of D does this program handle?
    d_start  = pid_d * BLOCK_D
    d_offs   = d_start + tl.arange(0, BLOCK_D)         # [BLOCK_D]
    d_mask   = d_offs < D

    # Look up token index for every row
    token_idx = tl.load(idx_ptr + row_idx, mask=row_mask, other=0)   # [BLOCK_N]

    # Load embedding row slice
    offs = token_idx[:, None] * D + d_offs[None, :]        # [BLOCK_N, BLOCK_D]
    mask = row_mask[:, None] & d_mask[None, :]

    emb = tl.load(emb_ptr + offs, mask=mask, other=0.0)    # [BLOCK_N, BLOCK_D]

    # Store to output
    tl.store(out_ptr + row_idx[:, None] * D + d_offs[None, :],
             emb, mask=mask)



# ---------------------------------------------------------------------------
# @torch.fx.wrap wrapper – called with (input_ids, weight) by the framework.
# D is pulled from weight.shape[1] at runtime (always 1536 for this graph).
# KEY DESIGN: allocate the output tensor with the CORRECT shape (*input_ids.
# shape, D) using only torch.empty (no reshape/view calls).  input_ids is
# treated as a flat 1-D array inside the kernel – its data pointer already
# points to the first int64.  The output tensor is already [B, S, D] so no
# view is needed on return.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _embedding_triton(input_ids, weight):
    # These are Python-level attribute accesses / arithmetic – no ATen ops
    seq_shape = input_ids.shape          # torch.Size – pure Python tuple-like
    D         = weight.shape[1]          # runtime embedding dim
    N         = 1
    for s in seq_shape:
        N *= s                            # total tokens (Python integer math)

    # torch.empty with *seq_shape + D is an allocation API (allowed)
    out = torch.empty(*seq_shape, D, dtype=weight.dtype, device=weight.device)

    grid = lambda meta: (
        triton.cdiv(N, meta['BLOCK_N']),
        triton.cdiv(D, meta['BLOCK_D']),
    )
    # input_ids.data_ptr() already points to a flat int64 array of N elements
    _embedding_fwd[grid](input_ids, weight, out, N, D)

    return out   # already the correct [B, S, D] shape







# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------
def pattern(input_ids, weight):
    emb     = torch.nn.functional.embedding(input_ids, weight, None, None, 2.0, False, False)
    return emb


def replacement_args(input_ids, weight):
    return (input_ids, weight)


def replacement_func():
    return _embedding_triton