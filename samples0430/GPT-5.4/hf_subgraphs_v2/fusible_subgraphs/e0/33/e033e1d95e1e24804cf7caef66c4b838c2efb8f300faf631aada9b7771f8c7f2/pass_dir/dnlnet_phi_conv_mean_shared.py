import torch
import triton
import triton.language as tl

OUT_C = 256
HW = 4096


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 64, "BLOCK_R": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_C": 64, "BLOCK_R": 256}, num_warps=4, num_stages=3),
    ],
    key=[],
)
@triton.jit
def theta_mean_kernel(
    inp_ptr,
    out_ptr,
    stride_ib,
    stride_ir,
    stride_ic,
    stride_ob,
    stride_oc,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_c = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < OUT_C

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for r0 in tl.static_range(0, HW, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        mask_r = offs_r < HW
        vals = tl.load(
            inp_ptr + pid_b * stride_ib + offs_r[:, None] * stride_ir + offs_c[None, :] * stride_ic,
            mask=mask_r[:, None] & mask_c[None, :],
            other=0.0,
        )
        acc += tl.sum(vals.to(tl.float32), axis=0)

    acc *= 1.0 / HW
    tl.store(out_ptr + pid_b * stride_ob + offs_c * stride_oc, acc, mask=mask_c)


@torch.fx.wrap
def dnlnet_phi_conv_mean(in_0, in_1, in_2, in_3):
    batch = in_3.shape[0]
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(batch, OUT_C, -1)

    tmp_4 = torch.empty((batch, 1, OUT_C), device=in_2.device, dtype=in_2.dtype)
    mean_grid = (triton.cdiv(OUT_C, 64), batch)
    theta_mean_kernel[mean_grid](
        in_2,
        tmp_4,
        in_2.stride(0),
        in_2.stride(1),
        in_2.stride(2),
        tmp_4.stride(0),
        tmp_4.stride(2),
    )
    return tmp_4, tmp_3


def replacement_func():
    return dnlnet_phi_conv_mean