import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: relu → 3× identical max_pool2d(kernel=5,stride=1,pad=2,dil=1) → cat
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return (tmp_4,)


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Fused Triton kernel:
#   - One program per (b, c) pair
#   - Loads H×W input elements, applies relu
#   - Computes 5×5 max-pool (padding=2, stride=1, dilation=1) in-register
#   - Writes 4 copies to output: [relu, pool, pool, pool] along channel dim
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
        triton.Config({'BLOCK_HW': 512}, num_warps=16),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=16),
    ],
    key=['H', 'W', 'C'],
)
@triton.jit
def relu_maxpool5_cat_kernel(
    in_ptr, out_ptr,
    B, C, H, W,
    BLOCK_HW: tl.constexpr,
):
    """
    Grid: (B * C,)  — one program per (batch, channel) pair.
    Each program handles H*W spatial positions.
    Output layout: [B, 4*C, H, W]
      - channels [0,   C):  relu(input)
      - channels [C,   2C): max_pool5(relu(input))
      - channels [2C,  3C): max_pool5(relu(input))   (same as above)
      - channels [3C,  4C): max_pool5(relu(input))   (same as above)
    """
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    base_in = b * C * H * W + c * H * W

    hw_ids = tl.arange(0, BLOCK_HW)
    mask = hw_ids < H * W

    # Safe h/w: use 0 for out-of-range lanes (they are never stored)
    h = tl.where(mask, hw_ids // W, 0)
    w = tl.where(mask, hw_ids % W, 0)

    # -----------------------------------------------------------------------
    # 1. Load and relu
    # -----------------------------------------------------------------------
    x = tl.load(in_ptr + base_in + hw_ids, mask=mask, other=0.0)
    relu_x = tl.maximum(x, 0.0)

    # -----------------------------------------------------------------------
    # 2. 5×5 max-pool on relu output
    #    - padding=2, stride=1, dilation=1  → output same spatial size as input
    #    - Since relu(x) >= 0, out-of-bound "pad" values are effectively 0,
    #      which is <= any valid relu value, so we can initialise with 0.
    # -----------------------------------------------------------------------
    pool_max = relu_x  # centre is always in-range → valid initialiser

    for dh in range(5):           # dh = 0..4  → offset = dh-2 ∈ [-2, 2]
        for dw in range(5):       # dw = 0..4  → offset = dw-2 ∈ [-2, 2]
            nh = h + dh - 2
            nw = w + dw - 2
            in_range = mask & (nh >= 0) & (nh < H) & (nw >= 0) & (nw < W)
            # Clamp indices so we never generate out-of-bounds pointers
            nh_c = tl.maximum(0, tl.minimum(nh, H - 1))
            nw_c = tl.maximum(0, tl.minimum(nw, W - 1))
            val = tl.load(in_ptr + base_in + nh_c * W + nw_c,
                          mask=mask, other=0.0)
            relu_val = tl.maximum(val, 0.0)
            # Keep max; for out-of-range neighbors use pool_max (no change)
            pool_max = tl.maximum(pool_max,
                                  tl.where(in_range, relu_val, pool_max))

    # -----------------------------------------------------------------------
    # 3. Write 4 channels to output
    # -----------------------------------------------------------------------
    CHW = C * H * W
    out_base = b * 4 * CHW

    # ch 0..C-1 : relu(input)
    tl.store(out_ptr + out_base + c * H * W + hw_ids,
             relu_x, mask=mask)
    # ch C..2C-1 : max_pool(relu(input))
    tl.store(out_ptr + out_base + (C + c) * H * W + hw_ids,
             pool_max, mask=mask)
    # ch 2C..3C-1
    tl.store(out_ptr + out_base + (2 * C + c) * H * W + hw_ids,
             pool_max, mask=mask)
    # ch 3C..4C-1
    tl.store(out_ptr + out_base + (3 * C + c) * H * W + hw_ids,
             pool_max, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be @torch.fx.wrap so FX doesn't trace into it)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def relu_maxpool5_cat(in_0):
    B, C, H, W = in_0.shape
    out = torch.empty((B, 4 * C, H, W), dtype=in_0.dtype, device=in_0.device)

    relu_maxpool5_cat_kernel[(B * C,)](
        in_0, out,
        B, C, H, W,
    )

    return (out,)


def replacement_func():
    return relu_maxpool5_cat