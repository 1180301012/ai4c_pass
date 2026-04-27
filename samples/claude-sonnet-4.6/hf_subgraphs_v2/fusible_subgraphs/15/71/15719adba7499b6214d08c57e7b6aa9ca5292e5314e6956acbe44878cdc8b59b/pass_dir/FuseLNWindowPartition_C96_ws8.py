"""
Optimization pass for Swin-tiny arocr patch embedding:
  layer_norm(C=96) + dropout(no-op)
  → Triton layer_norm kernel

Graph shape context:
  tmp_7 input:  [1, 65536, 96]  NON-CONTIGUOUS (strides [6291456, 1, 65536])
                                because it is the result of transpose(1,2).
  out_ln output: [1, 65536, 96] CONTIGUOUS — matches PyTorch layer_norm behaviour.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: layer norm over the last dimension (C=96)
# Each program handles one row (one patch).
# Input must be CONTIGUOUS (caller ensures this).
# BLOCK_C must be >= C and a power of 2; 128 is the minimum for C=96.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_C": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_C": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 128}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_C": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_C": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 256}, num_warps=16, num_stages=1),
    ],
    key=["C"],
)
@triton.jit
def _ln_fwd_c96(
    X,              # input pointer (possibly non-contiguous)
    W,              # scale pointer (contiguous [C])
    B,              # bias pointer  (contiguous [C])
    Y,              # output pointer (contiguous [N, C])
    N,              # number of rows
    C,              # number of cols (= 96)
    stride_xrow,    # X stride along the row (patch) dimension
    stride_xcol,    # X stride along the col (channel) dimension
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)

    cols = tl.arange(0, BLOCK_C)
    mask = cols < C

    # Load using actual strides — handles non-contiguous (transposed) input
    x = tl.load(X + row * stride_xrow + cols * stride_xcol,
                 mask=mask, other=0.0).to(tl.float32)

    # Mean — padded lanes are 0 and don't bias the sum
    mean = tl.sum(x, axis=0) / C

    # Centre; explicitly zero padded lanes before variance computation
    xc = tl.where(mask, x - mean, 0.0)

    # Variance
    var = tl.sum(xc * xc, axis=0) / C

    # Normalise
    xn = xc * tl.rsqrt(var + eps)

    # Scale & shift
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xn * w + b

    # Store to contiguous output (Triton casts fp32 → output dtype)
    tl.store(Y + row * C + cols, y, mask=mask)


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fused_ln_c96(tmp_7, weight, bias):
    """
    Args
        tmp_7  : [1, 65536, 96]  possibly non-contiguous transposed conv output
        weight : [96]            LN scale  (in_2)
        bias   : [96]            LN bias   (in_1)
    Returns
        out_ln : [1, 65536, 96]  CONTIGUOUS layer-norm result (== tmp_9)
    """
    N_patches = 65536   # 256*256 patches after stride-4 conv on 1024x1024
    C = 96

    # Read actual strides (Python int, not a torch op — safe under API restrictions)
    stride_row = tmp_7.stride(1)   # patch dim
    stride_col = tmp_7.stride(2)   # channel dim

    # Allocate contiguous output (matches PyTorch layer_norm output layout)
    out_ln = torch.empty(1, N_patches, C, dtype=tmp_7.dtype, device=tmp_7.device)

    _ln_fwd_c96[(N_patches,)](
        tmp_7, weight, bias, out_ln,
        N=N_patches,
        C=C,
        stride_xrow=stride_row,
        stride_xcol=stride_col,
        eps=1e-5,
    )

    return out_ln


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------
def pattern(tmp_7, in_2, in_1):
    """
    Matches layer-norm + no-op dropout for swin_arocr_tiny (C=96).
    """
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (96,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    return tmp_9


def replacement_args(tmp_7, in_2, in_1):
    return (tmp_7, in_2, in_1)


def replacement_func():
    return _fused_ln_c96