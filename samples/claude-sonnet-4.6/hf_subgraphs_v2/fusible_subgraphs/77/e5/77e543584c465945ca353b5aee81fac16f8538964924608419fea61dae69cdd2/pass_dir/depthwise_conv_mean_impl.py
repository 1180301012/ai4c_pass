"""
Shared Triton kernel: fused depthwise-conv2d (3×3, pad=1) + spatial mean.

Grid  : (N * C,)   — one program per (batch, channel)
Tile  : BLOCK_W elements across W_out (vectorised)
Loop  : iterates over all H_out rows; 3×3 kernel unrolled statically

Benefit over separate conv + mean:
  the mean accumulation happens while writing the conv output,
  so we make only one read of the (large) output tensor instead of two.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_W": 64}, num_warps=4),
        triton.Config({"BLOCK_W": 64}, num_warps=8),
    ],
    key=["H_out", "W_out", "C"],
)
@triton.jit
def _dw_conv3x3_mean_kernel(
    inp_ptr,    # [N, C, H, W]
    wt_ptr,     # [C, 9]  (reshaped from [C, 1, 3, 3])
    out_ptr,    # [N, C, H_out, W_out]
    mean_ptr,   # [N, C]  (will be reshaped to [N,C,1,1] in wrapper)
    C,
    H, W,
    H_out, W_out,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    inp_N_s, inp_C_s, inp_H_s,   # input strides (W stride = 1)
    out_N_s, out_C_s, out_H_s,   # output strides (W stride = 1)
    IS_FP16: tl.constexpr,        # True when dtype is float16
    IS_BF16: tl.constexpr,        # True when dtype is bfloat16
    BLOCK_W: tl.constexpr,        # must be >= W_out (we tile W fully in one pass)
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    # base pointers for this (n, c) slice
    inp_nc = inp_ptr + n * inp_N_s + c * inp_C_s
    out_nc = out_ptr + n * out_N_s + c * out_C_s
    wt_c   = wt_ptr  + c * 9

    ow_range = tl.arange(0, BLOCK_W)
    valid_w  = ow_range < W_out

    # Vector accumulator for spatial mean: shape [BLOCK_W]
    mean_sum = tl.zeros([BLOCK_W], dtype=tl.float32)

    for oh in range(H_out):
        row_acc = tl.zeros([BLOCK_W], dtype=tl.float32)

        # Unroll the 3×3 kernel
        for kh in tl.static_range(3):
            ih = oh * stride_h + kh - 1          # may be -1 or H on boundary

            for kw in tl.static_range(3):
                iw = ow_range * stride_w + kw - 1  # shape [BLOCK_W]

                valid = valid_w & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

                inp_val = tl.load(
                    inp_nc + ih * inp_H_s + iw,
                    mask=valid, other=0.0,
                )
                w_val = tl.load(wt_c + kh * 3 + kw)

                row_acc = row_acc + inp_val.to(tl.float32) * w_val.to(tl.float32)

        # Write output row — cast back to original dtype
        if IS_FP16:
            out_row = row_acc.to(tl.float16)
        elif IS_BF16:
            out_row = row_acc.to(tl.bfloat16)
        else:
            out_row = row_acc  # float32

        tl.store(out_nc + oh * out_H_s + ow_range, out_row, mask=valid_w)

        # Accumulate into spatial mean (mask out-of-range lanes)
        mean_sum = mean_sum + tl.where(valid_w, row_acc, tl.zeros([BLOCK_W], dtype=tl.float32))

    # Reduce across W and divide to get mean value
    mean_f32 = tl.sum(mean_sum) / (H_out * W_out)

    # Cast and store
    if IS_FP16:
        mean_val = mean_f32.to(tl.float16)
    elif IS_BF16:
        mean_val = mean_f32.to(tl.bfloat16)
    else:
        mean_val = mean_f32
    tl.store(mean_ptr + n * C + c, mean_val)


# ---------------------------------------------------------------------------
# Python helper (not @torch.fx.wrap — called inside the fx.wrap dispatcher)
# ---------------------------------------------------------------------------

def depthwise_conv3x3_mean_fused(inp: torch.Tensor,
                                  wt: torch.Tensor,
                                  stride: int) -> tuple:
    """
    Parameters
    ----------
    inp    : [N, C, H, W]  on CUDA
    wt     : [C, 1, 3, 3]  may be on CPU
    stride : 1 or 2

    Returns
    -------
    (out, mean)  where out : [N, C, H_out, W_out]
                          mean: [N, C, 1, 1]
    both on the same device/dtype as inp
    """
    N, C, H, W = inp.shape

    # PyTorch conv output size formula (pad=1, dil=1, k=3)
    H_out = (H + 2 - 3) // stride + 1
    W_out = (W + 2 - 3) // stride + 1

    # Weight: move to GPU and reshape to [C, 9] for easy indexing
    wt_dev = wt.to(device=inp.device, dtype=inp.dtype).reshape(C, 9).contiguous()

    inp_c = inp.contiguous()

    out     = torch.empty((N, C, H_out, W_out), dtype=inp.dtype, device=inp.device)
    mean_nc = torch.empty((N, C),               dtype=inp.dtype, device=inp.device)

    # Strides for contiguous NCHW tensors
    inp_N_s = C * H * W
    inp_C_s = H * W
    inp_H_s = W

    out_N_s = C * H_out * W_out
    out_C_s = H_out * W_out
    out_H_s = W_out

    IS_FP16 = (inp.dtype == torch.float16)
    IS_BF16 = (inp.dtype == torch.bfloat16)

    grid = (N * C,)

    _dw_conv3x3_mean_kernel[grid](
        inp_c, wt_dev, out, mean_nc,
        C,
        H, W,
        H_out, W_out,
        stride, stride,
        inp_N_s, inp_C_s, inp_H_s,
        out_N_s, out_C_s, out_H_s,
        IS_FP16, IS_BF16,
    )

    return out, mean_nc.reshape(N, C, 1, 1)


# ---------------------------------------------------------------------------
# Shared @torch.fx.wrap dispatcher — returned by replacement_func() in all
# pass files so that replacement_func_limit never drops any pass.
#
# Each pass differentiates itself by the route string appended to
# replacement_args(), which controls which stride is used.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fast mean-over-HW Triton kernel (single-output, no multi-output FX issues)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 512}, num_warps=8),
        triton.Config({"BLOCK": 1024}, num_warps=8),
    ],
    key=["HW"],
)
@triton.jit
def _mean_hw_kernel(
    inp_ptr, out_ptr,
    HW,
    inp_NC_s,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    inp_base = inp_ptr + pid * inp_NC_s
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, HW, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < HW
        vals = tl.load(inp_base + offs, mask=mask, other=0.0)
        acc = acc + vals.to(tl.float32)
    total = tl.sum(acc)
    mean = total / HW
    if IS_FP16:
        tl.store(out_ptr + pid, mean.to(tl.float16))
    elif IS_BF16:
        tl.store(out_ptr + pid, mean.to(tl.bfloat16))
    else:
        tl.store(out_ptr + pid, mean)


@torch.fx.wrap
def fast_mean_hw(x):
    """Single-output replacement for x.mean((2,3), keepdim=True)."""
    xc = x.contiguous()
    N, C, H, W = xc.shape
    NC = N * C
    HW = H * W
    IS_FP16 = (xc.dtype == torch.float16)
    IS_BF16 = (xc.dtype == torch.bfloat16)
    out_flat = torch.empty((NC,), dtype=xc.dtype, device=xc.device)
    _mean_hw_kernel[(NC,)](
        xc, out_flat,
        HW,
        H * W,       # inp_NC_s (contiguous)
        IS_FP16, IS_BF16,
    )
    return out_flat.reshape(N, C, 1, 1)
# A single @torch.fx.wrap returning a tuple is ONE node, so multi-output
# substitution fails.  We therefore split into two @torch.fx.wrap functions
# that share a tiny thread-local cache so the fused kernel runs only once.
# ---------------------------------------------------------------------------

# Module-level caches (one element each; populated by _get_conv_*, consumed
# by the following _get_mean_* call in the SAME forward pass).
_s1_cached_mean = None
_s2_cached_mean = None


@torch.fx.wrap
def _get_conv_s1(in_0, in_1):
    """Stride-1: run fused kernel, cache mean, return conv output."""
    global _s1_cached_mean
    conv_out, mean_out = depthwise_conv3x3_mean_fused(in_1, in_0, 1)
    _s1_cached_mean = mean_out
    return conv_out


@torch.fx.wrap
def _get_mean_s1(in_0, in_1):
    """Stride-1: retrieve cached mean (or recompute as fallback)."""
    global _s1_cached_mean
    if _s1_cached_mean is not None:
        mean_out = _s1_cached_mean
        _s1_cached_mean = None
        return mean_out
    _, mean_out = depthwise_conv3x3_mean_fused(in_1, in_0, 1)
    return mean_out


@torch.fx.wrap
def _get_conv_s2(in_0, in_1):
    """Stride-2: run fused kernel, cache mean, return conv output."""
    global _s2_cached_mean
    conv_out, mean_out = depthwise_conv3x3_mean_fused(in_1, in_0, 2)
    _s2_cached_mean = mean_out
    return conv_out


@torch.fx.wrap
def _get_mean_s2(in_0, in_1):
    """Stride-2: retrieve cached mean (or recompute as fallback)."""
    global _s2_cached_mean
    if _s2_cached_mean is not None:
        mean_out = _s2_cached_mean
        _s2_cached_mean = None
        return mean_out
    _, mean_out = depthwise_conv3x3_mean_fused(in_1, in_0, 2)
    return mean_out


def _replacement_fn_s1(in_0, in_1):
    """
    Traceable (NOT @torch.fx.wrap) replacement for stride-1 patterns.
    FX sees TWO call nodes → maps correctly to (conv2d, mean) outputs.
    """
    conv = _get_conv_s1(in_0, in_1)
    mean = _get_mean_s1(in_0, in_1)
    return conv, mean


def _replacement_fn_s2(in_0, in_1):
    """Traceable replacement for stride-2 patterns."""
    conv = _get_conv_s2(in_0, in_1)
    mean = _get_mean_s2(in_0, in_1)
    return conv, mean