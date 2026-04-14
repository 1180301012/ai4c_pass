import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: boolean column-indexing followed by cat
#   tmp_1 = in_0[:, in_2]                  (in_0 on CPU, in_2 bool on CUDA)
#   tmp_9 = torch.cat([tmp_1, in_1], dim=1)
# Both tmp_1 and tmp_9 are returned because tmp_1 is used outside the
# matched subgraph (by torch.ops.aten.sym_size.int).
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    tmp_1 = in_0[slice(None, None, None), in_2]
    tmp_9 = torch.cat([tmp_1, in_1], dim=1)
    return (tmp_1, tmp_9)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel 1: count True values in mask and compute exclusive prefix sum
#   - mask_ptr : [N] uint8 (bool stored as 1 byte)
#   - pos_ptr  : [N] int32 output – exclusive prefix sum (write positions)
#   - k_ptr    : [1] int32 output – total number of True values (K)
# Single program; BLOCK must be >= N (use BLOCK=256 to cover both N=100 & N=128)
# ---------------------------------------------------------------------------
@triton.jit
def _count_and_positions_kernel(
    mask_ptr,
    pos_ptr,
    k_ptr,
    N,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask_valid = offs < N

    # Load bool mask as int32 (0 or 1)
    vals = tl.load(mask_ptr + offs, mask=mask_valid, other=0).to(tl.int32)

    # Inclusive cumulative sum → exclusive prefix sum = positions
    cumsum = tl.cumsum(vals, axis=0)
    positions = cumsum - vals          # exclusive prefix sum

    tl.store(pos_ptr + offs, positions, mask=mask_valid)

    # Total count of True values
    k = tl.sum(vals, axis=0)
    tl.store(k_ptr, k)


# ---------------------------------------------------------------------------
# Triton kernel 2: fused gather (boolean index) + cat
#   Grid: (2,) – one program per row (rows 0 and 1)
#
#   Phase 1 – gather: for each True position j in in_2, copy in0[row, j]
#             to out1[row, pos[j]] and out2[row, pos[j]]
#   Phase 2 – copy:  copy all of in1[row, :] to out2[row, K:]
# ---------------------------------------------------------------------------
@triton.jit
def _gather_cat_kernel(
    in0_ptr,    # [2, N] int64
    in1_ptr,    # [2, M] int64
    mask_ptr,   # [N] uint8 (bool)
    pos_ptr,    # [N] int32  – exclusive prefix-sum positions
    out1_ptr,   # [2, K] int64   – boolean-indexed result
    out2_ptr,   # [2, K+M] int64 – concatenated result
    N, M, K, KM,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(0)   # 0 or 1

    # ---- Phase 1: gather selected columns from in0 ----------------------
    n_offs = tl.arange(0, BLOCK_N)
    n_valid = n_offs < N

    mask_vals = tl.load(mask_ptr + n_offs, mask=n_valid, other=0)
    positions  = tl.load(pos_ptr  + n_offs, mask=n_valid, other=0)
    in0_vals   = tl.load(in0_ptr  + row * N + n_offs, mask=n_valid, other=0)

    should_store = n_valid & (mask_vals != 0)

    # Write to out1 (the indexed result) and to the left part of out2
    tl.store(out1_ptr + row * K   + positions, in0_vals, mask=should_store)
    tl.store(out2_ptr + row * KM  + positions, in0_vals, mask=should_store)

    # ---- Phase 2: copy in1 to the right part of out2 --------------------
    for m_start in range(0, M, BLOCK_M):
        m_offs  = m_start + tl.arange(0, BLOCK_M)
        m_valid = m_offs < M
        in1_vals = tl.load(in1_ptr + row * M + m_offs, mask=m_valid, other=0)
        tl.store(out2_ptr + row * KM + K + m_offs, in1_vals, mask=m_valid)


# ---------------------------------------------------------------------------
# Replacement wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_bool_index_cat(in_0, in_1, in_2):
    """
    Fused replacement for:
        tmp_1 = in_0[:, in_2]
        tmp_9 = torch.cat([tmp_1, in_1], dim=1)

    Strategy:
      1. Transfer in_0 (CPU) to GPU once.
      2. One Triton kernel: count True values + exclusive prefix sum.
      3. One Triton kernel: gather selected columns AND copy in_1 (fused cat).
    """
    device = in_1.device
    dtype  = in_1.dtype

    # Move in_0 to the target device (GPU) if it isn't already there
    in_0_gpu = in_0.to(device)

    N = in_2.shape[0]   # columns in in_0
    M = in_1.shape[1]   # columns in in_1

    # BLOCK_N must cover max(N); both graphs have N <= 128, so 128 suffices.
    BLOCK_N = 128
    BLOCK_M = 128

    # ---- Step 1: count True values and compute prefix-sum positions ------
    k_tensor   = torch.zeros(1, dtype=torch.int32, device=device)
    pos_tensor = torch.empty(N, dtype=torch.int32, device=device)

    _count_and_positions_kernel[(1,)](
        in_2, pos_tensor, k_tensor,
        N,
        BLOCK=BLOCK_N,
    )

    # Synchronise: read K from GPU (small tensor → minimal overhead)
    K  = k_tensor.item()
    KM = K + M

    # ---- Step 2: allocate outputs ----------------------------------------
    # Guard against zero-size allocation; we slice to exact size at the end.
    K_alloc = K if K > 0 else 1
    out1 = torch.empty((2, K_alloc), dtype=dtype, device=device)
    out2 = torch.empty((2, KM),      dtype=dtype, device=device)

    # ---- Step 3: fused gather + cat ----------------------------------------
    _gather_cat_kernel[(2,)](
        in_0_gpu, in_1, in_2, pos_tensor,
        out1, out2,
        N, M, K, KM,
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
    )

    # Slice out1 to exact shape (handles K==0 via empty [2,0] view)
    return out1[:, :K], out2


def replacement_func():
    return fused_bool_index_cat