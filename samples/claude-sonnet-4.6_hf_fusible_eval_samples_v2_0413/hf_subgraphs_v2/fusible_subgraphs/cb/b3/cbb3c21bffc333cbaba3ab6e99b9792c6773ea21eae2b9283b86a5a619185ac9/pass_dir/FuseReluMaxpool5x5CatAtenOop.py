"""
Aten-operator variant with OUT-OF-PLACE relu.
Some torch.export decompositions replace relu_(inplace) with relu(out-of-place).
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_W': 64}, num_warps=4, num_stages=2),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def _relu_mp5_cat_v2_kernel(
    inp_ptr,
    out_ptr,
    N, C, H, W,
    BLOCK_W: tl.constexpr,
):
    pid    = tl.program_id(0)
    h_out  = pid % H
    c      = (pid // H) % C
    n      = pid // (C * H)

    w_offs = tl.arange(0, BLOCK_W)
    mask_w = w_offs < W

    inp_nc_base = inp_ptr + n * C * H * W + c * H * W

    inp_row  = tl.load(inp_nc_base + h_out * W + w_offs, mask=mask_w, other=0.0)
    zero_v   = tl.zeros([BLOCK_W], dtype=inp_row.dtype)
    relu_row = tl.maximum(inp_row, zero_v)

    out_relu = out_ptr + n * (4 * C * H * W) + c * (H * W) + h_out * W
    tl.store(out_relu + w_offs, relu_row, mask=mask_w)

    acc = tl.zeros([BLOCK_W], dtype=inp_row.dtype)

    for dh in tl.static_range(5):
        h_in      = h_out + dh - 2
        h_valid   = (h_in >= 0) & (h_in < H)
        h_in_safe = tl.where(h_valid, h_in, 0)

        for dw in tl.static_range(5):
            w_in      = w_offs + (dw - 2)
            w_valid   = (w_in >= 0) & (w_in < W)
            w_in_safe = tl.where(w_valid, w_in, 0)
            combined  = mask_w & w_valid & h_valid

            val = tl.load(inp_nc_base + h_in_safe * W + w_in_safe,
                          mask=combined, other=0.0)
            acc = tl.maximum(acc, val)

    base = out_ptr + n * (4 * C * H * W) + h_out * W
    tl.store(base + (    C + c) * (H * W) + w_offs, acc, mask=mask_w)
    tl.store(base + (2 * C + c) * (H * W) + w_offs, acc, mask=mask_w)
    tl.store(base + (3 * C + c) * (H * W) + w_offs, acc, mask=mask_w)


@torch.fx.wrap
def _relu_mp5_cat_fused_v2(in_0):
    N, C, H, W = in_0.shape
    out = torch.empty((N, 4 * C, H, W), dtype=in_0.dtype, device=in_0.device)
    _relu_mp5_cat_v2_kernel[(N * C * H,)](in_0, out, N, C, H, W)
    return out


# ---------------------------------------------------------------------------
# Pattern  (aten operators, OUT-OF-PLACE relu)
# ---------------------------------------------------------------------------

def pattern(in_0):
    tmp_0 = torch.ops.aten.relu.default(in_0)
    tmp_1 = torch.ops.aten.max_pool2d.default(tmp_0, [5, 5], [1, 1], [2, 2], [1, 1], False)
    tmp_2 = torch.ops.aten.max_pool2d.default(tmp_0, [5, 5], [1, 1], [2, 2], [1, 1], False)
    tmp_3 = torch.ops.aten.max_pool2d.default(tmp_0, [5, 5], [1, 1], [2, 2], [1, 1], False)
    tmp_4 = torch.ops.aten.cat.default([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return _relu_mp5_cat_fused_v2