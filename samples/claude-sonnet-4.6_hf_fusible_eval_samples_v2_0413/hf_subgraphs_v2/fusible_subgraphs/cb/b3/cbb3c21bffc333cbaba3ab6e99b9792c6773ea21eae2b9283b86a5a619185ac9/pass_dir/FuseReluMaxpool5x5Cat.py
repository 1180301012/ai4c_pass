import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 32}, num_warps=4, num_stages=3),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def relu_maxpool5x5_cat_kernel(
    inp_ptr,
    out_ptr,
    N, C, H, W,
    BLOCK_W: tl.constexpr,
):
    """
    Fused kernel: relu + max_pool2d(k=5,s=1,p=2,d=1) x3 + cat([relu, pool, pool, pool], dim=1)

    Input:  [N, C, H, W]
    Output: [N, 4*C, H, W]
      channels [0   :  C) = relu(input)
      channels [C   : 2C) = max_pool(relu(input))  -- same as next two
      channels [2C  : 3C) = max_pool(relu(input))
      channels [3C  : 4C) = max_pool(relu(input))

    One program handles all W elements of one (n, c, h_out) row.
    Grid: (N * C * H,)
    """
    pid = tl.program_id(0)

    # Decompose pid -> (n, c, h_out)
    h_out = pid % H
    c     = (pid // H) % C
    n     = pid // (C * H)

    w_offs = tl.arange(0, BLOCK_W)
    mask_w = w_offs < W

    # Base pointer for input[n, c, :, :]
    inp_nc_base = inp_ptr + n * C * H * W + c * H * W

    # ------------------------------------------------------------------
    # Load input row, apply relu, write to output channel c
    # ------------------------------------------------------------------
    inp_row = tl.load(inp_nc_base + h_out * W + w_offs, mask=mask_w, other=0.0)
    zero_vec = tl.zeros([BLOCK_W], dtype=inp_row.dtype)
    relu_row = tl.maximum(inp_row, zero_vec)

    out_relu_ptr = out_ptr + n * (4 * C * H * W) + c * (H * W) + h_out * W
    tl.store(out_relu_ptr + w_offs, relu_row, mask=mask_w)

    # ------------------------------------------------------------------
    # Compute max_pool2d(relu(input)) over 5x5 window (stride=1, pad=2)
    #
    # Key insight: max(relu(x_i)) = max(0, x_1, ..., x_n)
    # because acc starts at 0 (the relu lower-bound) and we take the max
    # with raw input values.  Out-of-bound loads return other=0.0 which
    # is correct (padding contributes 0 = relu(0) to the max).
    # ------------------------------------------------------------------
    acc = tl.zeros([BLOCK_W], dtype=inp_row.dtype)

    for dh in tl.static_range(5):          # actual offset = dh - 2 = -2..2
        h_in = h_out + dh - 2              # Triton scalar (0-D tensor)
        h_valid = (h_in >= 0) & (h_in < H)
        h_in_safe = tl.where(h_valid, h_in, 0)

        for dw in tl.static_range(5):      # actual offset = dw - 2 = -2..2
            w_in = w_offs + (dw - 2)       # 1-D vector
            w_valid = (w_in >= 0) & (w_in < W)
            w_in_safe = tl.where(w_valid, w_in, 0)

            combined_valid = mask_w & w_valid & h_valid

            val = tl.load(
                inp_nc_base + h_in_safe * W + w_in_safe,
                mask=combined_valid,
                other=0.0,
            )
            acc = tl.maximum(acc, val)

    # ------------------------------------------------------------------
    # Write pooled result to channels C+c, 2C+c, 3C+c  (3 copies)
    # ------------------------------------------------------------------
    base_pool = out_ptr + n * (4 * C * H * W) + h_out * W
    tl.store(base_pool + (    C + c) * (H * W) + w_offs, acc, mask=mask_w)
    tl.store(base_pool + (2 * C + c) * (H * W) + w_offs, acc, mask=mask_w)
    tl.store(base_pool + (3 * C + c) * (H * W) + w_offs, acc, mask=mask_w)


@torch.fx.wrap
def relu_maxpool_cat_fused(in_0):
    N, C, H, W = in_0.shape
    out = torch.empty((N, 4 * C, H, W), dtype=in_0.dtype, device=in_0.device)

    total_programs = N * C * H   # one program per (batch, channel, row)

    relu_maxpool5x5_cat_kernel[(total_programs,)](
        in_0, out,
        N, C, H, W,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return relu_maxpool_cat_fused