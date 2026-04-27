"""
Optimization pass for Swinv2 patch embedding:
  layer_norm(C=16) + dropout(no-op)
  → Triton layer_norm kernel

Graph shape context:
  tmp_7 input:  [1, 256, 16]  NON-CONTIGUOUS (strides [4096, 1, 256])
                              because it is the result of transpose(1,2).
  out_ln output: [1, 256, 16] CONTIGUOUS — matches PyTorch layer_norm behaviour.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: layer norm over the last dimension (C=16)
# Each program handles one row (one patch).
# Input is CONTIGUOUS (caller must ensure this).
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 16}, num_warps=1),
        triton.Config({"BLOCK_C": 32}, num_warps=2),
    ],
    key=["C"],
)
@triton.jit
def _ln_fwd_c16(
    X,              # input pointer (possibly non-contiguous)
    W,              # scale pointer (contiguous [C])
    B,              # bias pointer  (contiguous [C])
    Y,              # output pointer (contiguous [N, C])
    N,              # number of rows
    C,              # number of cols (= 16)
    stride_xrow,    # X stride along the row (patch) dimension
    stride_xcol,    # X stride along the col (channel) dimension
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)

    cols = tl.arange(0, BLOCK_C)
    mask = cols < C

    # Load using the actual strides — handles non-contiguous (transposed) input
    x = tl.load(X + row * stride_xrow + cols * stride_xcol,
                 mask=mask, other=0.0).to(tl.float32)

    # Mean
    mean = tl.sum(x, axis=0) / C

    # Centre; zero padded lanes
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
# Kernel wrapper (must be @torch.fx.wrap so FX doesn't trace inside)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fused_ln_c16(tmp_7, weight, bias):
    """
    Args
        tmp_7  : [1, 256, 16]  possibly non-contiguous transposed conv output
        weight : [16]          LN scale  (in_2)
        bias   : [16]          LN bias   (in_1)
    Returns
        out_ln : [1, 256, 16]  CONTIGUOUS layer-norm result (== tmp_9)
    """
    N_patches = 256   # 16*16 patches after stride-2 conv on 32x32
    C = 16

    # Read actual strides (Python int, not a torch op — safe under API restrictions)
    stride_row = tmp_7.stride(1)   # patch dim
    stride_col = tmp_7.stride(2)   # channel dim

    # Allocate contiguous output (matches PyTorch layer_norm output layout)
    out_ln = torch.empty(1, N_patches, C, dtype=tmp_7.dtype, device=tmp_7.device)

    _ln_fwd_c16[(N_patches,)](
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
    Matches layer-norm + no-op dropout for Swinv2 tiny (C=16).
    """
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (16,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    return tmp_9


def replacement_args(tmp_7, in_2, in_1):
    return (tmp_7, in_2, in_1)


def replacement_func():
    return _fused_ln_c16