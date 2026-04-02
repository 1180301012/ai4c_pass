import torch
import operator
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fallback: all high-level Python ops (F.relu instead of aten.relu)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    tmp_5 = torch.nn.functional.max_pool2d(in_3, 2, 1, 0, 1, ceil_mode=True, return_indices=False)
    tmp_6 = torch.cat([tmp_5, tmp_4], dim=1)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Triton kernel – fused (same as FuseFullAtenDecomposed but with list args)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=4),
    ],
    key=['B', 'C', 'H', 'W'],
)
@triton.jit
def fused_relu_scale_maxpool_cat_kernel2(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr, out_ptr,
    B, C, H, W,
    in2_stride_b, in2_stride_c, in2_stride_h,
    in3_stride_b, in3_stride_c, in3_stride_h,
    out_stride_b, out_stride_c, out_stride_h,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    N       = B * (2 * C) * H * W
    mask    = offsets < N

    scale = tl.load(in1_ptr)
    bias  = tl.load(in0_ptr)

    w_idx = offsets % W
    h_idx = (offsets // W) % H
    c_idx = (offsets // (W * H)) % (2 * C)
    b_idx = offsets // (W * H * 2 * C)

    is_pool = c_idx < C
    is_ew   = c_idx >= C

    pool_c    = tl.where(is_pool, c_idx, 0)
    pool_base = (b_idx * in3_stride_b + pool_c * in3_stride_c +
                 h_idx * in3_stride_h + w_idx)
    v00 = tl.load(in3_ptr + pool_base,                    mask=mask & is_pool, other=float('-inf'))
    v01 = tl.load(in3_ptr + pool_base + 1,                mask=mask & is_pool, other=float('-inf'))
    v10 = tl.load(in3_ptr + pool_base + in3_stride_h,     mask=mask & is_pool, other=float('-inf'))
    v11 = tl.load(in3_ptr + pool_base + in3_stride_h + 1, mask=mask & is_pool, other=float('-inf'))
    pool_val = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))

    ew_c    = tl.where(is_pool, 0, c_idx - C)
    ew_base = (b_idx * in2_stride_b + ew_c * in2_stride_c +
               h_idx * in2_stride_h + w_idx)
    in2_val  = tl.load(in2_ptr + ew_base, mask=mask & is_ew, other=0.0)
    relu_val = tl.maximum(in2_val, 0.0)
    ew_val   = relu_val * scale + bias

    out_val  = tl.where(is_pool, pool_val, ew_val)
    out_base = (b_idx * out_stride_b + c_idx * out_stride_c +
                h_idx * out_stride_h + w_idx)
    tl.store(out_ptr + out_base, out_val, mask=mask)


@torch.fx.wrap
def fused_relu_scale_maxpool_cat2(in_0, in_1, in_2, in_3):
    B, C, H, W = in_2.shape
    out = torch.empty(B, 2 * C, H, W, dtype=in_2.dtype, device=in_2.device)
    N   = B * 2 * C * H * W

    in2_stride_b, in2_stride_c, in2_stride_h, _ = in_2.stride()
    in3_stride_b, in3_stride_c, in3_stride_h, _ = in_3.stride()
    out_stride_b, out_stride_c, out_stride_h, _ = out.stride()

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    fused_relu_scale_maxpool_cat_kernel2[grid](
        in_0, in_1, in_2, in_3, out,
        B, C, H, W,
        in2_stride_b, in2_stride_c, in2_stride_h,
        in3_stride_b, in3_stride_c, in3_stride_h,
        out_stride_b, out_stride_c, out_stride_h,
    )
    return out


def replacement_func():
    return fused_relu_scale_maxpool_cat2