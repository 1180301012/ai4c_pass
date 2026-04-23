import torch
import triton
import triton.language as tl


OUT_LEN = 8400


@triton.jit
def yolo11l_obb_prefix_kernel(
    in3_ptr,
    in4_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    pid = tl.program_id(1)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < 8000

    is_left = offs < 6400

    left_vals = tl.load(
        in3_ptr + batch * 6400 + offs,
        mask=mask & is_left,
        other=0.0,
    ).to(tl.float32)
    right_vals = tl.load(
        in4_ptr + batch * 1600 + (offs - 6400),
        mask=mask & (~is_left),
        other=0.0,
    ).to(tl.float32)

    x = left_vals + right_vals
    y = (1.0 / (1.0 + tl.exp(-x)) - 0.25) * 3.141592653589793

    tl.store(out_ptr + batch * 8400 + offs, y, mask=mask)


@triton.jit
def yolo11l_obb_conv_tail_kernel(
    bias_ptr,
    weight_ptr,
    in2_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0)
    pid = tl.program_id(1)

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < 400

    acc = tl.full((BLOCK_M,), 0.0, tl.float32)
    bias = tl.load(bias_ptr).to(tl.float32)
    acc += bias

    for k in range(0, 64, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < 64
        w = tl.load(weight_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        x = tl.load(
            in2_ptr + batch * 25600 + offs_k[None, :] * 400 + offs_m[:, None],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(x * w[None, :], axis=1)

    y = (1.0 / (1.0 + tl.exp(-acc)) - 0.25) * 3.141592653589793
    tl.store(out_ptr + batch * 8400 + 8000 + offs_m, y, mask=mask_m)


@torch.fx.wrap
def yolo11l_obb_start587_end593_fused(in_0, in_1, in_2, in_3, in_4):
    batch = in_2.shape[0]
    out = torch.empty((batch, 1, OUT_LEN), device=in_2.device, dtype=in_2.dtype)

    yolo11l_obb_prefix_kernel[(batch, triton.cdiv(8000, 1024))](
        in_3,
        in_4,
        out,
        BLOCK_SIZE=1024,
        num_warps=4,
        num_stages=2,
    )

    yolo11l_obb_conv_tail_kernel[(batch, triton.cdiv(400, 128))](
        in_0,
        in_1,
        in_2,
        out,
        BLOCK_M=128,
        BLOCK_K=16,
        num_warps=4,
        num_stages=2,
    )

    return out