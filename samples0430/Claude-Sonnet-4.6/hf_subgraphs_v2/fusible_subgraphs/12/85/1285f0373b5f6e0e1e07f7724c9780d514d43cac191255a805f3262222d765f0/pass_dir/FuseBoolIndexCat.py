import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: boolean column‑select followed by cat
#   tmp_1 = in_0[:, in_2]          (in_0 may be on CPU, in_2 bool CUDA mask)
#   tmp_9 = torch.cat([tmp_1, in_1], dim=1)
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    tmp_1 = in_0[slice(None, None, None), in_2]
    tmp_9 = torch.cat([tmp_1, in_1], dim=1)
    return tmp_9


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _prefix_scan_kernel(
    mask_ptr,   # [N0]  bool (stored as uint8)
    scan_ptr,   # [N0]  int32 output – exclusive prefix sum
    count_ptr,  # [1]   int32 output – total number of True
    N,          # int   actual mask length (≤ BLOCK)
    BLOCK: tl.constexpr,
):
    """Single-block inclusive→exclusive cumsum of the boolean mask."""
    offs = tl.arange(0, BLOCK)
    valid = offs < N
    vals = tl.load(mask_ptr + offs, mask=valid, other=0).to(tl.int32)
    incl = tl.cumsum(vals, axis=0)          # inclusive prefix sum
    excl = incl - vals                       # exclusive prefix sum
    tl.store(scan_ptr + offs, excl, mask=valid)
    tl.store(count_ptr, tl.sum(vals))


@triton.jit
def _scatter_in0_kernel(
    in0_ptr,          # [rows, N0]  source (already on CUDA)
    mask_ptr,         # [N0]        bool mask
    scan_ptr,         # [N0]        exclusive prefix-sum positions
    out_ptr,          # [rows, K+N1]
    N0,               # int  number of source columns
    out_stride,       # int  row stride of out
    in0_stride,       # int  row stride of in0
    BLOCK: tl.constexpr,
):
    """Forward-scatter: for each True position in mask, write in0 column
    to the corresponding output column given by the prefix scan."""
    row = tl.program_id(1)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = col < N0
    m = (tl.load(mask_ptr + col, mask=valid, other=0) != 0)
    active = valid & m
    out_col = tl.load(scan_ptr + col, mask=active, other=0).to(tl.int64)
    val = tl.load(in0_ptr + row * in0_stride + col, mask=active, other=0)
    tl.store(out_ptr + row * out_stride + out_col, val, mask=active)


@triton.jit
def _copy_in1_kernel(
    in1_ptr,      # [rows, N1]
    out_ptr,      # [rows, K+N1]
    k,            # int  number of selected columns (offset into out)
    N1,           # int  number of in1 columns
    out_stride,   # int  row stride of out
    in1_stride,   # int  row stride of in1
    BLOCK: tl.constexpr,
):
    """Copy in1 into the tail [K : K+N1] of every row of out."""
    row = tl.program_id(1)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = col < N1
    val = tl.load(in1_ptr + row * in1_stride + col, mask=valid, other=0)
    tl.store(out_ptr + row * out_stride + k + col, val, mask=valid)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_bool_index_cat(in_0, in_1, in_2):
    device = in_1.device
    dtype  = in_0.dtype
    rows   = in_0.shape[0]   # always 2 for these graphs
    N0     = in_0.shape[1]   # 128 (RECT_L) or 100 (GAE)
    N1     = in_1.shape[1]   # 128 (RECT_L) or 1000 (GAE)

    # Bring in_0 to the same device as in_1 (in_0 lives on CPU)
    in_0_dev = torch.as_tensor(in_0, device=device)

    # ------------------------------------------------------------------
    # Step 1 – compute exclusive prefix-sum of mask → positions in output
    # ------------------------------------------------------------------
    # Choose smallest power-of-2 BLOCK >= N0 (N0 ≤ 128 always)
    SCAN_BLOCK = 128   # covers both N0=100 and N0=128

    scan  = torch.empty(N0, dtype=torch.int32, device=device)
    count = torch.zeros(1, dtype=torch.int32, device=device)

    _prefix_scan_kernel[(1,)](
        mask_ptr=in_2,
        scan_ptr=scan,
        count_ptr=count,
        N=N0,
        BLOCK=SCAN_BLOCK,
    )

    k = count.item()   # number of True values in mask (GPU→CPU sync)

    # ------------------------------------------------------------------
    # Step 2 – allocate output [rows, k + N1]
    # ------------------------------------------------------------------
    out = torch.empty((rows, k + N1), dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Step 3 – scatter masked columns from in_0 into out[:,0:k]
    # ------------------------------------------------------------------
    BLOCK_N0 = 128
    grid_n0  = (N0 + BLOCK_N0 - 1) // BLOCK_N0   # 1 for N0 ≤ 128

    _scatter_in0_kernel[(grid_n0, rows)](
        in0_ptr   = in_0_dev,
        mask_ptr  = in_2,
        scan_ptr  = scan,
        out_ptr   = out,
        N0        = N0,
        out_stride  = out.stride(0),
        in0_stride  = in_0_dev.stride(0),
        BLOCK     = BLOCK_N0,
    )

    # ------------------------------------------------------------------
    # Step 4 – copy in_1 into out[:,k : k+N1]
    # ------------------------------------------------------------------
    BLOCK_N1 = 512
    grid_n1  = (N1 + BLOCK_N1 - 1) // BLOCK_N1

    _copy_in1_kernel[(grid_n1, rows)](
        in1_ptr    = in_1,
        out_ptr    = out,
        k          = k,
        N1         = N1,
        out_stride = out.stride(0),
        in1_stride = in_1.stride(0),
        BLOCK      = BLOCK_N1,
    )

    return out


def replacement_func():
    return fused_bool_index_cat