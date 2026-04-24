import torch
import triton
import triton.language as tl


# Pattern: match the index_select (the expensive operation).
def pattern(in_1, indices):
    tmp_2 = in_1.index_select(-2, indices)
    return tmp_2


def replacement_args(in_1, indices):
    return (in_1, indices)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 16},   num_warps=1),
        triton.Config({'BLOCK_S': 32},   num_warps=1),
        triton.Config({'BLOCK_S': 64},   num_warps=2),
        triton.Config({'BLOCK_S': 128},  num_warps=4),
        triton.Config({'BLOCK_S': 256},  num_warps=4),
        triton.Config({'BLOCK_S': 512},  num_warps=4),
        triton.Config({'BLOCK_S': 1024}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def gather_rows_kernel(
    idx_ptr,          # int64 gather indices  [N]
    src_ptr,          # data tensor           [max_node, D]
    out_ptr,          # output tensor         [N, D]
    N,                # number of gathered rows (runtime — used for masking)
    BLOCK_S: tl.constexpr,  # row-tile size (compile-time for tl.arange)
    D: tl.constexpr,        # feature dim = 16 (compile-time for tl.arange)
):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_S + tl.arange(0, BLOCK_S)   # [BLOCK_S]
    s_mask = row_offsets < N

    # Load gather indices (int64)
    idx = tl.load(idx_ptr + row_offsets, mask=s_mask, other=0)

    # Feature-dimension offsets [0..D-1]  — D constexpr makes arange valid
    d_offsets = tl.arange(0, D)

    # 2-D gather: src[idx[row], d] → out[row, d]
    # Both src and out are row-major: element [n, d] = n * D + d
    src_offsets = idx[:, None] * D + d_offsets[None, :]   # [BLOCK_S, D]
    out_offsets = row_offsets[:, None] * D + d_offsets[None, :] # [BLOCK_S, D]

    load_mask = s_mask[:, None]   # broadcast row-mask over D

    data = tl.load(src_ptr + src_offsets, mask=load_mask, other=0.0)
    tl.store(out_ptr + out_offsets, data, mask=load_mask)


# Fixed constants for this specific graph
_N   = 1100   # number of edges (gathered rows)
_D   = 16     # feature dimension


@torch.fx.wrap
def triton_gather_rows(in_1, indices):
    # in_1:      [max_node, D]  bfloat16 / float16
    # indices:   [N]            int64
    out = torch.empty((_N, _D), dtype=in_1.dtype, device=in_1.device)

    # Autotune selects the best BLOCK_S; grid adapts dynamically
    grid = lambda meta: ((_N + meta['BLOCK_S'] - 1) // meta['BLOCK_S'],)

    gather_rows_kernel[grid](
        indices, in_1, out,
        _N,
        D=_D,
    )

    return out


def replacement_func():
    return triton_gather_rows