import torch
import triton
import triton.language as tl


def pattern(x):
    return torch.nn.functional.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)


def replacement_args(x):
    return (x,)


@triton.jit
def bilinear_upsample_kernel(
    input_ptr, output_ptr,
    C, H_IN, W_IN, H_OUT, W_OUT,
    HW_IN, HW_OUT,
    TILE_HW_OUT: tl.constexpr, TILE_C: tl.constexpr,
    DTYPE_FLAG: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)

    hw_out_start = pid_hw * TILE_HW_OUT
    c_start = pid_c * TILE_C

    hw_out_range = hw_out_start + tl.arange(0, TILE_HW_OUT)
    c_range = c_start + tl.arange(0, TILE_C)

    # Convert flat output index to (oy, ox)
    oy = hw_out_range // W_OUT
    ox = hw_out_range % W_OUT

    # Source coordinates (align_corners=False)
    scale_y = H_IN * 1.0 / H_OUT
    scale_x = W_IN * 1.0 / W_OUT
    src_y = (oy + 0.5) * scale_y - 0.5
    src_x = (ox + 0.5) * scale_x - 0.5

    y0 = tl.maximum(tl.floor(src_y).to(tl.int32), 0)
    y1 = tl.minimum(y0 + 1, H_IN - 1)
    x0 = tl.maximum(tl.floor(src_x).to(tl.int32), 0)
    x1 = tl.minimum(x0 + 1, W_IN - 1)

    # Interpolation weights
    wy1 = src_y - y0.to(tl.float32)
    wy0 = 1.0 - wy1
    wx1 = src_x - x0.to(tl.float32)
    wx0 = 1.0 - wx1

    # Masks
    mask_c = c_range < C
    mask_hw = hw_out_range < HW_OUT
    mask_2d = mask_c[:, None] & mask_hw[None, :]

    # Load 4 source corners - each load covers (TILE_C channels, TILE_HW_OUT spatial positions)
    # input layout: [C, H_IN * W_IN] for N=1 batch element
    # offset for (c, y, x) = c * HW_IN + y * W_IN + x
    v00 = tl.load(input_ptr + c_range[:, None] * HW_IN + y0[None, :] * W_IN + x0[None, :], mask=mask_2d, other=0.0).to(tl.float32)
    v01 = tl.load(input_ptr + c_range[:, None] * HW_IN + y0[None, :] * W_IN + x1[None, :], mask=mask_2d, other=0.0).to(tl.float32)
    v10 = tl.load(input_ptr + c_range[:, None] * HW_IN + y1[None, :] * W_IN + x0[None, :], mask=mask_2d, other=0.0).to(tl.float32)
    v11 = tl.load(input_ptr + c_range[:, None] * HW_IN + y1[None, :] * W_IN + x1[None, :], mask=mask_2d, other=0.0).to(tl.float32)

    # Bilinear interpolation in float32 for precision
    result = (wy0[None, :] * wx0[None, :] * v00 +
              wy0[None, :] * wx1[None, :] * v01 +
              wy1[None, :] * wx0[None, :] * v10 +
              wy1[None, :] * wx1[None, :] * v11)

    # Store with appropriate dtype conversion
    offsets_out = c_range[:, None] * HW_OUT + hw_out_range[None, :]
    if DTYPE_FLAG == 1:
        tl.store(output_ptr + offsets_out, result.to(tl.bfloat16), mask=mask_2d)
    else:
        tl.store(output_ptr + offsets_out, result.to(tl.float16), mask=mask_2d)


@torch.fx.wrap
def triton_bilinear_upsample(x):
    x = x.contiguous()
    N, C, H_IN, W_IN = x.shape
    H_OUT = 512
    W_OUT = 512
    HW_IN = H_IN * W_IN
    HW_OUT = H_OUT * W_OUT

    output = torch.empty((N, C, H_OUT, W_OUT), dtype=x.dtype, device=x.device)

    is_bf16 = (x.dtype == torch.bfloat16)
    DTYPE_FLAG = 1 if is_bf16 else 0

    TILE_HW_OUT = 256
    TILE_C = 32

    num_hw_tiles = triton.cdiv(HW_OUT, TILE_HW_OUT)
    num_c_tiles = triton.cdiv(C, TILE_C)

    grid = (num_hw_tiles, num_c_tiles)

    for n in range(N):
        bilinear_upsample_kernel[grid](
            input_ptr=x[n],
            output_ptr=output[n],
            C=C, H_IN=H_IN, W_IN=W_IN, H_OUT=H_OUT, W_OUT=W_OUT,
            HW_IN=HW_IN, HW_OUT=HW_OUT,
            TILE_HW_OUT=TILE_HW_OUT, TILE_C=TILE_C,
            DTYPE_FLAG=DTYPE_FLAG,
        )

    return output


def replacement_func():
    return triton_bilinear_upsample