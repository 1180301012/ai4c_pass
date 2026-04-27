"""
Shared Triton layer-norm dispatch for 768 and 1024 variants.

Key design notes
----------------
• tmp_7 = (conv2d+in_4).flatten(2).transpose(1,2) has shape [1,256,C]
  with strides [256*C, 1, 256] — NON-CONTIGUOUS in the last dim.
• Kernels hard-code all dimensions (stride_row=1, stride_col=256, N_rows=256)
  to eliminate every Python-level PosionDispatch call inside _do_ln_*.
• Output is CONTIGUOUS [1,256,C] via torch.empty() to avoid aten.empty_like
  dispatch and to get coalesced writes.
• Both pass files import _triton_ln_dispatch from here so they share the
  EXACT SAME Python object → satisfies the g_replacement_func singleton.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel for C=768  (input strides: row=1, col=256; output: contiguous)
# ---------------------------------------------------------------------------

@triton.jit
def _ln_768_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BLOCK_C: tl.constexpr,
):
    # shape [1,256,768], strides [196608,1,256]
    # row index = sequence position (0..255)
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < 768

    # Non-contiguous load: row*1 + col*256
    x_offs = row + cols * 256
    safe_x = tl.minimum(x_offs, 196607)   # clamp OOB masked accesses
    x = tl.load(X_ptr + safe_x, mask=mask, other=0.0).to(tl.float32)

    mean  = tl.sum(x, 0) / 768
    diff  = tl.where(mask, x - mean, 0.0)
    var   = tl.sum(diff * diff, 0) / 768
    rstd  = tl.rsqrt(var + 1e-5)
    xhat  = diff * rstd

    safe_w = tl.minimum(cols, 767)
    w = tl.load(W_ptr + safe_w, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + safe_w, mask=mask, other=0.0).to(tl.float32)
    y = xhat * w + b

    # Contiguous store: row*768 + col
    tl.store(Y_ptr + row * 768 + cols, y, mask=mask)


# ---------------------------------------------------------------------------
# Kernel for C=1024  (input strides: row=1, col=256; output: contiguous)
# ---------------------------------------------------------------------------

@triton.jit
def _ln_1024_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BLOCK_C: tl.constexpr,
):
    # shape [1,256,1024], strides [262144,1,256]
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < 1024   # always True when BLOCK_C == 1024

    x_offs = row + cols * 256
    # max valid: 255 + 1023*256 = 255 + 261888 = 262143
    x = tl.load(X_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

    mean  = tl.sum(x, 0) / 1024
    diff  = tl.where(mask, x - mean, 0.0)
    var   = tl.sum(diff * diff, 0) / 1024
    rstd  = tl.rsqrt(var + 1e-5)
    xhat  = diff * rstd

    # safe_w not needed: BLOCK_C==C==1024, no out-of-bounds for W/B
    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xhat * w + b

    tl.store(Y_ptr + row * 1024 + cols, y, mask=mask)


# ---------------------------------------------------------------------------
# Route helpers  (plain, NOT @torch.fx.wrap)
# Minimise Python overhead: no x.stride(), x.shape, x.numel() calls;
# torch.empty avoids aten.empty_like dispatch through PosionDispatchTensor.
# x.dtype / x.device are C++-level property reads (no __torch_dispatch__).
# ---------------------------------------------------------------------------

def _do_ln_768(x, weight, bias):
    y = torch.empty(1, 256, 768, dtype=x.dtype, device=x.device)
    _ln_768_kernel[(256,)](x, weight, bias, y, BLOCK_C=1024, num_warps=8)
    return y


def _do_ln_1024(x, weight, bias):
    y = torch.empty(1, 256, 1024, dtype=x.dtype, device=x.device)
    _ln_1024_kernel[(256,)](x, weight, bias, y, BLOCK_C=1024, num_warps=8)
    return y


# ---------------------------------------------------------------------------
# THE shared @torch.fx.wrap dispatch wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def _triton_ln_dispatch(x, weight, bias, route):
    """Opaque FX leaf: run layer-norm via Triton (768 or 1024 channels)."""
    if route == "768":
        return _do_ln_768(x, weight, bias)
    else:
        return _do_ln_1024(x, weight, bias)