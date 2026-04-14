import torch
import triton
import triton.language as tl


def pattern(tmp_6, tmp_7, in_1, in_0):
    tmp_8 = tmp_6 + tmp_7
    tmp_9 = tmp_8.transpose(1, 2)
    tmp_10 = torch.nn.functional.layer_norm(tmp_9, (768,), in_1, in_0, 1e-05)
    return tmp_10


def replacement_args(tmp_6, tmp_7, in_1, in_0):
    return (tmp_6, tmp_7, in_1, in_0)


# ---------------------------------------------------------------------------
# Kernel 1: Fused add + tiled transpose  [C, T] → [T, C]
#
# Uses tl.make_block_ptr + tl.trans so that both reads (along T, stride=1)
# and writes (along C, stride=1) are coalesced.
# ---------------------------------------------------------------------------
@triton.jit
def _add_transpose_kernel(
    a_ptr, b_ptr, out_ptr,
    C, T,
    stride_a_c, stride_a_t,
    stride_b_c, stride_b_t,
    IS_BF16: tl.constexpr,
    TILE_C: tl.constexpr,
    TILE_T: tl.constexpr,
):
    """
    Reads a tile [TILE_C, TILE_T] from the [C, T] input tensors (coalesced
    along T), computes element-wise add, transposes the tile and writes it
    to the [T, C] output (coalesced along C).
    """
    c_block = tl.program_id(0)
    t_block = tl.program_id(1)

    c_start = c_block * TILE_C
    t_start = t_block * TILE_T

    # Block pointers – T is the fast (stride-1) dimension → coalesced loads
    a_bptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(C, T),
        strides=(stride_a_c, stride_a_t),
        offsets=(c_start, t_start),
        block_shape=(TILE_C, TILE_T),
        order=(1, 0),   # T is innermost → coalesced
    )
    b_bptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(C, T),
        strides=(stride_b_c, stride_b_t),
        offsets=(c_start, t_start),
        block_shape=(TILE_C, TILE_T),
        order=(1, 0),
    )

    a_tile = tl.load(a_bptr, boundary_check=(0, 1), padding_option='zero').to(tl.float32)
    b_tile = tl.load(b_bptr, boundary_check=(0, 1), padding_option='zero').to(tl.float32)

    sum_tile = a_tile + b_tile          # [TILE_C, TILE_T]
    sum_tile_t = tl.trans(sum_tile)     # [TILE_T, TILE_C]

    # Output block pointer – C is the fast (stride-1) dimension → coalesced stores
    out_bptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(T, C),
        strides=(C, 1),
        offsets=(t_start, c_start),
        block_shape=(TILE_T, TILE_C),
        order=(1, 0),   # C is innermost → coalesced
    )

    if IS_BF16:
        tl.store(out_bptr, sum_tile_t.to(tl.bfloat16), boundary_check=(0, 1))
    else:
        tl.store(out_bptr, sum_tile_t.to(tl.float16), boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# Kernel 2: Layer norm on a contiguous [N_ROWS, N_COLS] tensor.
#
# Each program handles one row.  All N_COLS elements of a row are
# contiguous → coalesced reads and writes.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['N_ROWS', 'N_COLS'],
)
@triton.jit
def _layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N_ROWS, N_COLS,
    inv_n_cols,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS

    # Contiguous read: x[row_idx, c]
    x_offs = row_idx * N_COLS + cols
    x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(mask, x, 0.0)

    mean = tl.sum(x, axis=0) * inv_n_cols
    x_c = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_c * x_c, axis=0) * inv_n_cols
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    x_norm = x_c * rstd

    w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out_f32 = x_norm * w + b

    out_offs = row_idx * N_COLS + cols
    if IS_BF16:
        tl.store(out_ptr + out_offs, out_f32.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + out_offs, out_f32.to(tl.float16), mask=mask)


# ---------------------------------------------------------------------------
# Replacement wrapper
# ---------------------------------------------------------------------------
TILE_C = 32
TILE_T = 32


@torch.fx.wrap
def fused_add_transpose_ln(tmp_6, tmp_7, in_1, in_0):
    """
    Replaces: add + transpose(1,2) + layer_norm(normalized_shape=(C,))

    tmp_6 : [B, C, T]  – contiguous  (e.g. [1, 768, 124])
    tmp_7 : [B, C, T]  – possibly non-contiguous (sliced from conv output)
    in_1  : [C]        – layer_norm weight
    in_0  : [C]        – layer_norm bias
    returns: [B, T, C]
    """
    B, C, T = tmp_6.shape
    IS_BF16 = (tmp_6.dtype == torch.bfloat16)

    # Step 1 – fused add + tiled transpose → contiguous [B, T, C]
    tmp_t = torch.empty((B, T, C), dtype=tmp_6.dtype, device=tmp_6.device)

    grid_c = (C + TILE_C - 1) // TILE_C
    grid_t = (T + TILE_T - 1) // TILE_T

    _add_transpose_kernel[(grid_c, grid_t)](
        tmp_6, tmp_7, tmp_t,
        C, T,
        tmp_6.stride(1), tmp_6.stride(2),   # stride_a_c, stride_a_t
        tmp_7.stride(1), tmp_7.stride(2),   # stride_b_c, stride_b_t
        IS_BF16,
        TILE_C, TILE_T,
    )

    # Step 2 – layer norm on contiguous rows of length C
    out = torch.empty((B, T, C), dtype=tmp_6.dtype, device=tmp_6.device)
    N_ROWS = B * T

    _layer_norm_kernel[(N_ROWS,)](
        tmp_t, in_1, in_0, out,
        N_ROWS, C,
        1.0 / C,
        IS_BF16,
    )
    return out


def replacement_func():
    return fused_add_transpose_ln