"""
Fused pass: add(in_4, in_5) + mean(dim=(2,3)) + dropout(noop) + dropout(noop) + batch_norm(inference)

Pattern input shapes:
  in_4, in_5: [N, C, H, W]   (float16 / bfloat16 / float32)
  in_0: running_mean [C]
  in_1: running_var  [C]
  in_2: bias         [C]
  in_3: weight       [C]

Returns: (bn_out [N, C], mean_out [N, C])

Observed HW values: 49 (7x7), 64 (8x8), 144 (12x12)
"""

import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────── #
#  Triton kernel – no autotune; BLOCK_HW set to next_power_of_2(HW).      #
#  Grid = (N*C,): one program per (n,c) pair, no sequential N loop.        #
#  num_warps=1 → A30 fits 32 blocks/SM → ~7 effective waves for N=32.     #
# ──────────────────────────────────────────────────────────────────────── #

@triton.jit
def fused_add_mean_bn_kernel(
    in4_ptr, in5_ptr,           # [N, C, H, W] contiguous
    rm_ptr, rv_ptr,             # running_mean, running_var  [C]
    w_ptr, b_ptr,               # weight, bias               [C]
    mean_out_ptr, bn_out_ptr,   # outputs  (row-major [N, C])
    C, HW, eps,
    BLOCK_HW: tl.constexpr,    # >= HW; next_power_of_2(HW)
):
    nc   = tl.program_id(0)
    c    = nc % C
    base = nc * HW

    offs  = tl.arange(0, BLOCK_HW)
    mask  = offs < HW

    x4 = tl.load(in4_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    x5 = tl.load(in5_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    total    = tl.sum(x4 + x5, axis=0)
    mean_val = total / HW

    rm      = tl.load(rm_ptr + c).to(tl.float32)
    rv      = tl.load(rv_ptr + c).to(tl.float32)
    w       = tl.load(w_ptr  + c).to(tl.float32)
    b       = tl.load(b_ptr  + c).to(tl.float32)
    inv_std = w / tl.sqrt(rv + eps)
    bn_val  = inv_std * mean_val + (b - inv_std * rm)

    tl.store(mean_out_ptr + nc, mean_val)
    tl.store(bn_out_ptr   + nc, bn_val)


# ──────────────────────────────────────────────────────────────────────── #
#  Pre-allocated output buffer cache; BLOCK_HW cached to avoid             #
#  triton.next_power_of_2() on every forward pass.                         #
# ──────────────────────────────────────────────────────────────────────── #
_OUTPUT_CACHE: dict = {}


@torch.fx.wrap
def _fused_add_mean_bn_inner(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Wrapped kernel launcher – FX treats this as a black box.
    in_0 = running_mean [C]
    in_1 = running_var  [C]
    in_2 = bias         [C]
    in_3 = weight       [C]
    in_4 = tensor A     [N, C, H, W]
    in_5 = tensor B     [N, C, H, W]

    Returns (bn_out, mean_out).
    """
    N, C, H, W = in_4.shape
    HW = H * W
    NC = N * C

    device_idx = in_4.device.index or 0
    cache_key  = (NC, HW, in_4.dtype, device_idx)
    entry = _OUTPUT_CACHE.get(cache_key)
    if entry is None:
        BLOCK_HW = max(triton.next_power_of_2(HW), 16)
        entry = (
            torch.empty((N, C), dtype=in_4.dtype, device=in_4.device),
            torch.empty((N, C), dtype=in_4.dtype, device=in_4.device),
            BLOCK_HW,
        )
        _OUTPUT_CACHE[cache_key] = entry
    mean_out, bn_out, BLOCK_HW = entry

    fused_add_mean_bn_kernel[(NC,)](
        in_4, in_5,
        in_0, in_1,   # rm, rv
        in_3, in_2,   # weight=in_3, bias=in_2
        mean_out, bn_out,
        C, HW, 1e-5,
        BLOCK_HW=BLOCK_HW,
        num_warps=1,
    )

    return bn_out, mean_out


def fused_add_mean_bn_inference(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    NOT wrapped – FX traces into this function so that the two outputs
    are represented as separate getitem nodes in the replacement graph.
    This ensures len(copied_returning_nodes) == 2, matching the two
    returning nodes (tmp_8, tmp_7) from the pattern.
    """
    result = _fused_add_mean_bn_inner(in_0, in_1, in_2, in_3, in_4, in_5)
    return result[0], result[1]


# ──────────────────────────────────────────────────────────────────────── #
#  Pattern / replacement interface                                          #
# ──────────────────────────────────────────────────────────────────────── #

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return (tmp_8, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return fused_add_mean_bn_inference