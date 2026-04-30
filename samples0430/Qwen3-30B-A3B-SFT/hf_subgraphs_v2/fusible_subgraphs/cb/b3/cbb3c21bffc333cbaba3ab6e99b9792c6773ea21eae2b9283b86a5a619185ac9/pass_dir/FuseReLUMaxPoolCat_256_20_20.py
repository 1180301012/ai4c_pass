import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match
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


# ---------------------------------------------------------------------------
# Fused Triton kernel
#   Grid: (N, cdiv(HW, BLOCK_HW))
#   Each program handles one (batch, hw-tile) pair, all 1024 output channels.
#
#   For each (n, hw_tile):
#     • Load relu values for C=256 channels at this spatial location.
#     • Compute 5×5 max-pool (pad=2, stride=1) on the relu values → C=256 pool values.
#     • Write to output:
#         out[n, 0:256,    h, w] = relu_val          (relu output)
#         out[n, 256:512,  h, w] = pool_val          (first pool)
#         out[n, 512:768,  h, w] = pool_val          (second pool – same)
#         out[n, 768:1024, h, w] = pool_val          (third pool)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 1024}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_HW": 512},  num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_HW": 256},  num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_HW": 256},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_HW": 128},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_HW": 64},   num_warps=4,  num_stages=2),
    ],
    key=["N", "C", "HW"],
)
@triton.jit
def fused_relu_maxpool_cat_kernel(
    in_ptr,      # [N, C, H, W]
    out_ptr,     # [N, 4*C, H, W]
    N, C, H, W, HW,
    BLOCK_HW: tl.constexpr,
):
    pid_n  = tl.program_id(0)
    pid_hw = tl.program_id(1)

    # Spatial offsets for this block
    hw_start   = pid_hw * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask    = hw_offsets < HW

    h_offs = hw_offsets // W
    w_offs = hw_offsets % W

    # Channel indices [0, 1, ..., 255]
    c_idx = tl.arange(0, 256)

    # Flat offset in [N, C, H, W]: n*C*HW + c*HW + hw
    base   = pid_n * C * HW
    in_off = base + c_idx[None, :] * HW + hw_offsets[:, None]   # [BLOCK_HW, 256]

    # ----- ReLU values -------------------------------------------------------
    x      = tl.load(in_ptr + in_off, mask=hw_mask[:, None], other=0.0)
    relu_v = tl.maximum(x, 0.0)                                  # [BLOCK_HW, 256]

    # ----- 5×5 max-pool (pad=2, stride=1, kernel=5) -------------------------
    pool_acc = tl.full([BLOCK_HW, 256], float("-inf"), dtype=tl.float32)

    for kh in tl.static_range(5):
        for kw in tl.static_range(5):
            ih = h_offs + (kh - 2)
            iw = w_offs + (kw - 2)
            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & hw_mask

            # Clamp to valid range so pointer arithmetic never goes out of bounds
            ih_c = tl.maximum(ih, 0)
            iw_c = tl.maximum(iw, 0)
            ih_c = tl.minimum(ih_c, H - 1)
            iw_c = tl.minimum(iw_c, W - 1)

            pool_off = base + c_idx[None, :] * HW + ih_c[:, None] * W + iw_c[:, None]
            pv = tl.load(in_ptr + pool_off, mask=valid[:, None], other=float("-inf"))
            pool_acc = tl.maximum(pool_acc, pv)

    pool_v = pool_acc.to(relu_v.dtype)   # [BLOCK_HW, 256]

    # ----- Concatenated output [N, 4*C, H, W] --------------------------------
    # out[n, c,       h, w] = relu_v       c in [0, 256)
    # out[n, 256+c,   h, w] = pool_v       c in [0, 256)
    # out[n, 512+c,   h, w] = pool_v       c in [0, 256)
    # out[n, 768+c,   h, w] = pool_v       c in [0, 256)
    out_base = pid_n * (4 * C * HW)

    c0       = c_idx[None, :]
    out_mask = hw_mask[:, None] & (c0 < 256)
    tl.store(out_ptr + out_base + c0,                         relu_v, mask=out_mask)

    c1       = tl.maximum(c_idx[None, :] - 256, 0)
    out_mask = hw_mask[:, None] & (c0 >= 256) & (c0 < 512)
    tl.store(out_ptr + out_base + 256 + c1,                   pool_v, mask=out_mask)

    c2       = tl.maximum(c_idx[None, :] - 512, 0)
    out_mask = hw_mask[:, None] & (c0 >= 512) & (c0 < 768)
    tl.store(out_ptr + out_base + 512 + c2,                   pool_v, mask=out_mask)

    c3       = tl.maximum(c_idx[None, :] - 768, 0)
    out_mask = hw_mask[:, None] & (c0 >= 768)
    tl.store(out_ptr + out_base + 768 + c3,                   pool_v, mask=out_mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_relu_maxpool_cat(in_0):
    N  = in_0.shape[0]
    C  = in_0.shape[1]
    H  = in_0.shape[2]
    W  = in_0.shape[3]
    HW = H * W

    out = torch.empty((N, 4 * C, H, W), dtype=in_0.dtype, device=in_0.device)

    def grid(meta):
        return (N, triton.cdiv(HW, meta["BLOCK_HW"]))

    fused_relu_maxpool_cat_kernel[grid](
        in_0, out,
        N, C, H, W, HW,
    )

    return out


def replacement_func():
    return fused_relu_maxpool_cat