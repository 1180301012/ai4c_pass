import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fuse the ENTIRE model forward in a single replacement:
#
#   conv2d(in_2, in_1, in_0) → view(1,2,8,8) → sigmoid  → tmp_4
#   in_3.sum(dim=3, keepdim=True)  →  in_3 / sum          → tmp_6
#   return (tmp_6, tmp_4)
#
# Shapes:
#   in_2 : [1,2,1,8]       – activation on CUDA
#   in_1 : [128,2,1,8]     – weight (128 output chans, 16 inner elements)
#   in_0 : [128]            – bias
#   in_3 : [1,2,8,8]       – 16 rows × 8 cols, sum/div along dim=3
#
# Two Triton kernels are launched from the single wrapper:
#   1. _conv_sigmoid_kernel  : coalesced 2D weight load, GEMV+sigmoid
#   2. _sum_div_kernel       : 2D block for row-wise normalization
# ---------------------------------------------------------------------------


def pattern(in_2, in_1, in_0, in_3):
    conv = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    y = conv.view(1, 2, 8, 8)
    z = y.sigmoid()
    s = in_3.sum(dim=3, keepdim=True)
    out = in_3 / s
    return (out, z)


def replacement_args(in_2, in_1, in_0, in_3):
    return (in_2, in_1, in_0, in_3)


# ---------------------------------------------------------------------------
# Kernel 1 – conv + view + sigmoid
#
# Weight layout:  in_1[o, j]  =  in1_ptr[o * N_IN + j]
# Each CTA processes BLOCK_OUT consecutive output channels, loading
# a contiguous [BLOCK_OUT × N_IN] block from the weight matrix.
#
# Grid: (N_OUT // BLOCK_OUT,)  =  (8,)   with BLOCK_OUT=16
# ---------------------------------------------------------------------------

@triton.jit
def _conv_sigmoid_kernel(
    in2_ptr,               # [N_IN]   = 16
    in1_ptr,               # [N_OUT, N_IN] = [128, 16]
    in0_ptr,               # [N_OUT]   = 128
    out_ptr,               # [N_OUT]   (maps to [1,2,8,8])
    N_IN:     tl.constexpr,   # 16
    N_OUT:    tl.constexpr,   # 128
    BLOCK_OUT: tl.constexpr,  # 16  →  8 CTAs
    IS_BF16:  tl.constexpr,
):
    pid     = tl.program_id(0)
    o_start = pid * BLOCK_OUT
    o_offs  = tl.arange(0, BLOCK_OUT)   # [16]
    j_offs  = tl.arange(0, N_IN)        # [16]

    # Load input vector (same for every CTA – hot in L1 after first CTA)
    x = tl.load(in2_ptr + j_offs).to(tl.float32)  # [N_IN]

    # Load weight block [BLOCK_OUT, N_IN] – contiguous in memory!
    # in1[(o_start + r), j] = in1_ptr[(o_start + r)*N_IN + j]
    # For fixed pid, (o_start + o_offs) × N_IN + j_offs is contiguous:
    #   rows [o_start..o_start+BLOCK_OUT-1], each of length N_IN
    w_flat_start = o_start * N_IN
    row_off = o_offs[:, None] * N_IN   # [BLOCK_OUT, 1]
    col_off = j_offs[None, :]          # [1, N_IN]
    w = tl.load(in1_ptr + w_flat_start + row_off + col_off).to(tl.float32)  # [BLOCK_OUT, N_IN]

    # Matrix-vector product: acc[r] = sum_j w[r,j] * x[j]
    acc = tl.sum(w * x[None, :], axis=1)  # [BLOCK_OUT]

    # Bias
    bias = tl.load(in0_ptr + o_start + o_offs).to(tl.float32)  # [BLOCK_OUT]
    acc  = acc + bias

    # Sigmoid
    result_f32 = 1.0 / (1.0 + tl.exp(-acc))

    if IS_BF16:
        result = result_f32.to(tl.bfloat16)
    else:
        result = result_f32.to(tl.float16)

    tl.store(out_ptr + o_start + o_offs, result)


# ---------------------------------------------------------------------------
# Kernel 2 – sum(dim=3, keepdim=True) + division
#
# Operates on x [1,2,8,8] viewed as [N_ROWS=16, W=8].
# Single CTA loads the full [16,8] block, computes per-row sums, divides.
#
# Grid: (1,)
# ---------------------------------------------------------------------------

@triton.jit
def _sum_div_kernel(
    x_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,  # 16
    W:      tl.constexpr,  # 8
    IS_BF16: tl.constexpr,
):
    row_offs = tl.arange(0, N_ROWS)   # [16]
    w_offs   = tl.arange(0, W)        # [8]

    # Build 2D flat indices [N_ROWS, W]
    flat_idx = row_offs[:, None] * W + w_offs[None, :]  # [16, 8]

    # Load entire block
    x = tl.load(x_ptr + flat_idx).to(tl.float32)   # [16, 8]

    # Per-row sum
    row_sum = tl.sum(x, axis=1)  # [16]

    # Normalize
    result_f32 = x / row_sum[:, None]  # [16, 8]

    if IS_BF16:
        result = result_f32.to(tl.bfloat16)
    else:
        result = result_f32.to(tl.float16)

    tl.store(out_ptr + flat_idx, result)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_all_ops(in_2, in_1, in_0, in_3):
    """
    Replaces:
        conv2d(in_2,in_1,in_0,(1,1),(0,0),(1,1),1).view(1,2,8,8).sigmoid()
        in_3 / in_3.sum(dim=3, keepdim=True)
    returning  (tmp_6, tmp_4)  exactly as the original model.
    """
    device  = in_2.device
    dtype   = in_2.dtype
    is_bf16 = (dtype == torch.bfloat16)

    # Move weight / bias to the compute device if needed
    in_1_dev = in_1.to(device)
    in_0_dev = in_0.to(device)

    N_IN      = 16   # 2 * 1 * 8
    N_OUT     = 128
    BLOCK_OUT = 16   # 8 coalesced CTAs for conv
    N_ROWS    = 16   # 1 * 2 * 8
    W         = 8

    # Output buffers
    out_z   = torch.empty((1, 2, 8, 8), dtype=dtype, device=device)  # sigmoid result
    out_div = torch.empty_like(in_3)                                   # normalized result

    # Kernel 1: conv + view + sigmoid
    _conv_sigmoid_kernel[(N_OUT // BLOCK_OUT,)](
        in_2.contiguous(),
        in_1_dev.contiguous(),
        in_0_dev.contiguous(),
        out_z,
        N_IN=N_IN,
        N_OUT=N_OUT,
        BLOCK_OUT=BLOCK_OUT,
        IS_BF16=is_bf16,
        num_warps=2,
    )

    # Kernel 2: row-wise normalization  (single CTA, 2D 16×8 block)
    _sum_div_kernel[(1,)](
        in_3.contiguous(),
        out_div,
        N_ROWS=N_ROWS,
        W=W,
        IS_BF16=is_bf16,
        num_warps=4,
    )

    # Return in the same order as the original model: (tmp_6, tmp_4)
    return (out_div, out_z)


def replacement_func():
    return fused_all_ops