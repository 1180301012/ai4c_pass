import torch
import triton
import triton.language as tl


def pattern(in_6):
    tmp_28 = in_6[(slice(None, None, None), slice(None, None, None), slice(1, -10, None), slice(None, None, None))]
    tmp_29 = tmp_28.transpose(2, 3)
    tmp_30 = tmp_29.view(4, 32, 15, 15)
    tmp_31 = torch.nn.functional.interpolate(tmp_30, size=(15, 15), mode='bicubic', align_corners=False)
    tmp_32 = tmp_31.flatten(2)
    tmp_33 = tmp_32.transpose(1, 2)
    tmp_34 = tmp_33.contiguous()
    tmp_35 = tmp_34.view(4, 1, 225, 32)
    return tmp_35


def replacement_args(in_6):
    return (in_6,)


@triton.jit
def yolos_mid_pos_copy_kernel(
    src_ptr,
    out_ptr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_t = pid_t * BT + tl.arange(0, BT)
    offs_c = tl.arange(0, BC)
    mask = (offs_t[:, None] < 225) & (offs_c[None, :] < 32)

    src_offsets = pid_b * 236 * 32 + (offs_t[:, None] + 1) * 32 + offs_c[None, :]
    out_offsets = pid_b * 225 * 32 + offs_t[:, None] * 32 + offs_c[None, :]

    x = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + out_offsets, x, mask=mask)


@torch.fx.wrap
def fused_yolos_embeddings(in_6):
    out = torch.empty((4, 1, 225, 32), device=in_6.device, dtype=in_6.dtype)
    BT = 32
    BC = 32
    grid = (triton.cdiv(225, BT), 4)
    yolos_mid_pos_copy_kernel[grid](
        in_6,
        out,
        BT=BT,
        BC=BC,
    )
    return out


def replacement_func():
    return fused_yolos_embeddings