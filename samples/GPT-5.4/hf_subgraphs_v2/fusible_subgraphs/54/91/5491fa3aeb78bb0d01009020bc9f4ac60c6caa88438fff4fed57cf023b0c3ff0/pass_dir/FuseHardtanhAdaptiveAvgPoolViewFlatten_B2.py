import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = tmp_1.view(2, -1)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_hardtanh_gap_kernel(
    x_ptr,
    out_ptr,
    hw,
    BLOCK_HW: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_HW)
    mask = offs < hw
    base = row * hw
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(tl.minimum(x, 6.0), 0.0)
    out = tl.sum(x, axis=0) / hw
    tl.store(out_ptr + row, out)


@torch.fx.wrap
def fused_hardtanh_gap(in_0):
    n = in_0.shape[0]
    c = in_0.shape[1]
    h = in_0.shape[2]
    w = in_0.shape[3]
    hw = h * w
    out = torch.empty((n, c), device=in_0.device, dtype=in_0.dtype)
    grid = (n * c,)

    if hw <= 16:
        fused_hardtanh_gap_kernel[grid](in_0, out, hw, BLOCK_HW=16, num_warps=1, num_stages=1)
    elif hw <= 64:
        fused_hardtanh_gap_kernel[grid](in_0, out, hw, BLOCK_HW=64, num_warps=1, num_stages=1)
    elif hw <= 128:
        fused_hardtanh_gap_kernel[grid](in_0, out, hw, BLOCK_HW=128, num_warps=2, num_stages=1)
    elif hw <= 256:
        fused_hardtanh_gap_kernel[grid](in_0, out, hw, BLOCK_HW=256, num_warps=4, num_stages=1)
    else:
        fused_hardtanh_gap_kernel[grid](in_0, out, hw, BLOCK_HW=512, num_warps=4, num_stages=1)

    return out


def replacement_func():
    return fused_hardtanh_gap