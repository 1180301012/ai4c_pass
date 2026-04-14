import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: BN (inference) + residual add + ReLU + spatial mean
# ---------------------------------------------------------------------------

def pattern(in_4, in_0, in_1, in_3, in_2, in_5):
    """
    Matches the exact dataflow from model.py:
        tmp_4 = batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
        tmp_5 = in_5 + tmp_4
        tmp_6 = relu(tmp_5, inplace=False)
        tmp_7 = tmp_6.mean((2, 3), keepdim=True)
        return (tmp_6, tmp_7)
    """
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_5 = in_5 + tmp_4
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
    tmp_7 = tmp_6.mean((2, 3), keepdim=True)
    return (tmp_6, tmp_7)


def replacement_args(in_4, in_0, in_1, in_3, in_2, in_5):
    return (in_4, in_0, in_1, in_3, in_2, in_5)


# ---------------------------------------------------------------------------
# Triton kernel: fused BN-inference + add + relu + mean(H,W)
# One program per (batch, channel) pair.
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    torch.bfloat16: 'bfloat16',
    torch.float16:  'float16',
    torch.float32:  'float32',
}


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128},  num_warps=4),
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=16),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def _bn_add_relu_mean_kernel(
    x_ptr,    # [N, C, H, W]  – BN input
    res_ptr,  # [N, C, H, W]  – residual to add
    rm_ptr,   # [C]            – running_mean
    rv_ptr,   # [C]            – running_var
    w_ptr,    # [C]            – BN weight (gamma)
    b_ptr,    # [C]            – BN bias  (beta)
    out_ptr,  # [N, C, H, W]  – relu output
    mo_ptr,   # [N, C, 1, 1]  – mean output (flat N*C)
    C,        # number of channels  (runtime)
    HW,       # H * W               (runtime)
    eps: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)          # pid ∈ [0, N*C)
    c   = pid % C

    # ---- per-channel BN parameters (loaded once, reused for all HW) --------
    rm = tl.load(rm_ptr + c).to(tl.float32)
    rv = tl.load(rv_ptr + c).to(tl.float32)
    wc = tl.load(w_ptr  + c).to(tl.float32)
    bc = tl.load(b_ptr  + c).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(rv + eps)
    scale   = wc * inv_std
    shift   = bc - rm * scale

    base = pid * HW
    acc  = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # ---- tile over spatial dimension ----------------------------------------
    for start in range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW

        x = tl.load(x_ptr   + base + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_ptr  + base + offs, mask=mask, other=0.0).to(tl.float32)

        y = x * scale + shift + r        # BN + residual add
        y = tl.where(y > 0.0, y, 0.0)   # ReLU

        # store relu output in original dtype
        if DTYPE == 'bfloat16':
            tl.store(out_ptr + base + offs, y.to(tl.bfloat16), mask=mask)
        elif DTYPE == 'float16':
            tl.store(out_ptr + base + offs, y.to(tl.float16),  mask=mask)
        else:
            tl.store(out_ptr + base + offs, y,                  mask=mask)

        acc = acc + tl.where(mask, y, 0.0)

    # ---- reduce to mean ------------------------------------------------------
    mean_val = tl.sum(acc) / HW          # float32 mean for this (n, c)

    if DTYPE == 'bfloat16':
        tl.store(mo_ptr + pid, mean_val.to(tl.bfloat16))
    elif DTYPE == 'float16':
        tl.store(mo_ptr + pid, mean_val.to(tl.float16))
    else:
        tl.store(mo_ptr + pid, mean_val)


# ---------------------------------------------------------------------------
# Wrapper called by the replacement
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_bn_add_relu_mean(in_4, in_0, in_1, in_3, in_2, in_5):
    """
    Arguments match replacement_args order:
        in_4  – BN input  [N, C, H, W]
        in_0  – running_mean [C]
        in_1  – running_var  [C]
        in_3  – weight (gamma) [C]
        in_2  – bias   (beta)  [C]
        in_5  – residual [N, C, H, W]
    Returns (relu_out, mean_out) matching model.py return (tmp_6, tmp_7).
    """
    N, C, H, W = in_4.shape
    HW = H * W
    dtype_str = _DTYPE_MAP[in_4.dtype]

    out      = torch.empty_like(in_4)
    mean_out = torch.empty(N, C, 1, 1, dtype=in_4.dtype, device=in_4.device)

    # One program per (batch, channel) pair
    _bn_add_relu_mean_kernel[(N * C,)](
        in_4, in_5,           # x, residual
        in_0, in_1, in_3, in_2,  # rm, rv, weight, bias
        out, mean_out,
        C, HW,
        1e-05,                # eps (constexpr)
        dtype_str,            # DTYPE (constexpr)
    )

    return out, mean_out


# ---------------------------------------------------------------------------
# replacement_func – returns the callable (do NOT call it here)
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_bn_add_relu_mean