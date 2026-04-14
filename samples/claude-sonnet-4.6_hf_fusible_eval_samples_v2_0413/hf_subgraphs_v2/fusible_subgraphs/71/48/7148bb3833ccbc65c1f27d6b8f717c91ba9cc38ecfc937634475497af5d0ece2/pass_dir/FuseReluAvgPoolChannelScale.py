import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match the post-pool computation using ONLY Python operators and
# tensor methods (no torch.nn.functional calls, which can't be traced).
#   pool_out  -> maps to tmp_3  = avg_pool2d(relu(in_2))
#   relu_out  -> maps to tmp_2  = relu(in_2)
#   in_0      -> maps to in_0   = channel scale weights
# ---------------------------------------------------------------------------
def pattern(pool_out, relu_out, in_0):
    tmp_4 = pool_out - relu_out
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.unsqueeze(-1)
    tmp_7 = tmp_6 * tmp_4
    tmp_8 = relu_out + tmp_7
    return tmp_8


def replacement_args(pool_out, relu_out, in_0):
    return (pool_out, relu_out, in_0)


# ---------------------------------------------------------------------------
# Triton kernel: out = relu_out + in_0[:,None,None] * (pool_out - relu_out)
# Tensors are contiguous → spatial offset = flat hw index (no h/w decomp).
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['B', 'C', 'H', 'W'],
)
@triton.jit
def scale_blend_kernel(
    pool_ptr, relu_ptr, scale_ptr, out_ptr,
    B, C, H, W,
    stride_b, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    bc_idx = tl.program_id(0)
    hw_block = tl.program_id(1)

    b_idx = bc_idx // C
    c_idx = bc_idx % C

    scale = tl.load(scale_ptr + c_idx).to(tl.float32)
    base = b_idx * stride_b + c_idx * stride_c

    hw_start = hw_block * BLOCK_SIZE
    hw_offs = hw_start + tl.arange(0, BLOCK_SIZE)
    mask = hw_offs < (H * W)

    # Contiguous layout: spatial offset = flat hw index
    pool_val = tl.load(pool_ptr + base + hw_offs, mask=mask, other=0.0)
    relu_val = tl.load(relu_ptr + base + hw_offs, mask=mask, other=0.0)

    pool_f32 = pool_val.to(tl.float32)
    relu_f32 = relu_val.to(tl.float32)

    result_f32 = relu_f32 + scale * (pool_f32 - relu_f32)
    result = result_f32.to(relu_val.dtype)

    tl.store(out_ptr + base + hw_offs, result, mask=mask)


@torch.fx.wrap
def fused_scale_blend(pool_out, relu_out, in_0):
    B, C, H, W = relu_out.shape
    out = torch.empty_like(relu_out)
    stride_b = C * H * W
    stride_c = H * W
    HW = H * W
    grid = lambda meta: (B * C, triton.cdiv(HW, meta['BLOCK_SIZE']))
    scale_blend_kernel[grid](
        pool_out, relu_out, in_0, out,
        B, C, H, W,
        stride_b, stride_c,
    )
    return out


def replacement_func():
    return fused_scale_blend