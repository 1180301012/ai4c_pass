import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: relu -> 3x identical max_pool2d(5,stride=1,pad=2,dil=1) -> cat
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
# Triton kernel: fused relu + 5x5 max-pool (computed ONCE) + cat (4 copies)
#
# Grid  : ceil(B*C*H*W / BLOCK_SIZE)  – flat 1-D over (b,c,h,w)
# Output: [B, 4C, H, W]
#   channels [0,   C):   relu(in)
#   channels [C,  2C):   max_pool2d(relu(in))
#   channels [2C, 3C):   same
#   channels [3C, 4C):   same
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['B', 'C', 'H', 'W'],
)
@triton.jit
def _fused_relu_pool_cat_kernel(
    in_ptr,
    out_ptr,
    B, C, H, W,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = B * C * H * W
    mask = offsets < total

    # ---- decode flat index -> (b, c, h, w) --------------------------------
    hw   = H * W
    chw  = C * hw

    hw_idx = offsets % hw
    c_idx  = (offsets // hw) % C
    b_idx  = offsets // chw

    h_idx = hw_idx // W
    w_idx = hw_idx % W

    # base address of this (b, c) slice in the input tensor
    in_base = b_idx * chw + c_idx * hw

    # ---- ReLU of the centre element ----------------------------------------
    center   = tl.load(in_ptr + in_base + h_idx * W + w_idx, mask=mask, other=0.0)
    relu_val = tl.maximum(center.to(tl.float32), 0.0)

    # ---- 5×5 max-pool over relu'd input, zero-padding at borders -----------
    # Initialise with 0.0 – correct because relu output ≥ 0 and padding is 0
    pool_max = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for kh in tl.static_range(5):
        for kw in tl.static_range(5):
            ih = h_idx + (kh - 2)
            iw = w_idx + (kw - 2)
            # validity mask for the pooling window (False → padding = 0)
            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            # clamp to a safe in-bounds address (actual load protected by valid)
            ih_c = tl.maximum(tl.minimum(ih, H - 1), 0)
            iw_c = tl.maximum(tl.minimum(iw, W - 1), 0)
            v = tl.load(
                in_ptr + in_base + ih_c * W + iw_c,
                mask=mask & valid,
                other=0.0,
            )
            v_relu = tl.maximum(v.to(tl.float32), 0.0)   # fused relu
            pool_max = tl.maximum(pool_max, v_relu)

    # ---- write to output [B, 4C, H, W] ------------------------------------
    C_hw   = C * hw                                        # stride of one channel group
    out_base = b_idx * (4 * C_hw) + c_idx * hw + h_idx * W + w_idx

    relu_out = relu_val.to(OUTPUT_DTYPE)
    pool_out = pool_max.to(OUTPUT_DTYPE)

    tl.store(out_ptr + out_base,             relu_out, mask=mask)   # channels [0, C)
    tl.store(out_ptr + out_base + C_hw,      pool_out, mask=mask)   # channels [C, 2C)
    tl.store(out_ptr + out_base + 2 * C_hw,  pool_out, mask=mask)   # channels [2C, 3C)
    tl.store(out_ptr + out_base + 3 * C_hw,  pool_out, mask=mask)   # channels [3C, 4C)


# ---------------------------------------------------------------------------
# Python-level wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_relu_pool_cat(in_0):
    B, C, H, W = in_0.shape
    out = torch.empty(B, 4 * C, H, W, dtype=in_0.dtype, device=in_0.device)
    total = B * C * H * W

    # Map PyTorch dtype -> Triton constexpr dtype
    dtype_map = {
        torch.float32:  tl.float32,
        torch.float16:  tl.float16,
        torch.bfloat16: tl.bfloat16,
    }
    output_dtype = dtype_map[in_0.dtype]

    def grid(meta):
        return ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _fused_relu_pool_cat_kernel[grid](
        in_0, out,
        B, C, H, W,
        OUTPUT_DTYPE=output_dtype,
    )
    return out


def replacement_func():
    return fused_relu_pool_cat