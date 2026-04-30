import torch
import triton
import triton.language as tl


PI = 3.141592653589793
SHIFT = 0.25
PREFIX_LEN = 8000
IN3_LEN = 6400
IN4_LEN = 1600
TAIL_LEN = 400
OUT_LEN = 8400
CHANNELS = 64
SPATIAL = 400


@triton.jit
def _prefix_sigmoid_affine_kernel(
    in3_ptr,
    in4_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    block_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    offs = block_id * BLOCK + tl.arange(0, BLOCK)
    mask = offs < 8000
    in3_mask = mask & (offs < 6400)
    in4_mask = mask & (offs >= 6400)

    offs4 = tl.where(offs >= 6400, offs - 6400, 0)

    x3 = tl.load(in3_ptr + batch_id * 6400 + offs, mask=in3_mask, other=0.0)
    x4 = tl.load(in4_ptr + batch_id * 1600 + offs4, mask=in4_mask, other=0.0)
    x = tl.where(offs < 6400, x3, x4).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-x))
    y = (sig - 0.25) * 3.141592653589793

    tl.store(out_ptr + batch_id * 8400 + offs, y, mask=mask)


@triton.jit
def _conv_tail_sigmoid_affine_kernel(
    in2_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    block_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    hw = block_id * BLOCK + tl.arange(0, BLOCK)
    mask = hw < 400

    acc = tl.load(bias_ptr).to(tl.float32)
    acc = acc + tl.zeros([BLOCK], dtype=tl.float32)

    batch_base = batch_id * 64 * 400

    for c in tl.static_range(0, 64):
        w = tl.load(weight_ptr + c).to(tl.float32)
        x = tl.load(in2_ptr + batch_base + c * 400 + hw, mask=mask, other=0.0).to(tl.float32)
        acc += x * w

    sig = 1.0 / (1.0 + tl.exp(-acc))
    y = (sig - 0.25) * 3.141592653589793

    tl.store(out_ptr + batch_id * 8400 + 8000 + hw, y, mask=mask)


@torch.fx.wrap
def fused_yolo_obb_conv_cat_sigmoid_sub_mul(in_0, in_1, in_2, in_3, in_4):
    batch = in_3.shape[0]
    out = torch.empty((batch, 1, OUT_LEN), device=in_3.device, dtype=in_3.dtype)

    grid_prefix = (triton.cdiv(PREFIX_LEN, 1024), batch)
    _prefix_sigmoid_affine_kernel[grid_prefix](
        in_3,
        in_4,
        out,
        BLOCK=1024,
        num_warps=8,
        num_stages=2,
    )

    grid_tail = (triton.cdiv(TAIL_LEN, 128), batch)
    _conv_tail_sigmoid_affine_kernel[grid_tail](
        in_2,
        in_1,
        in_0,
        out,
        BLOCK=128,
        num_warps=4,
        num_stages=2,
    )

    return out