import torch
import triton
import triton.language as tl


def pattern(x):
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    return (x,)


@triton.jit
def triton_mean_kernel(
    x_ptr, out_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel) slice
    pid = tl.program_id(0)
    base = pid * HW

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for i in range(0, tl.cdiv(HW, BLOCK_SIZE)):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW
        # Load input, accumulate in float32 for precision
        x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
        acc = acc + x.to(tl.float32)

    total = tl.sum(acc, axis=0)
    mean_val = total / HW
    # Store - Triton auto-converts float32 to output pointer's dtype
    tl.store(out_ptr + pid, mean_val)


@torch.fx.wrap
def triton_mean_2d(x):
    B, C, H, W = x.shape
    HW = H * W
    BC = B * C

    out = torch.empty((B, C, 1, 1), dtype=x.dtype, device=x.device)

    # Fixed block size; next power-of-2 >= HW so we do one iteration
    BLOCK_SIZE = max(256, triton.next_power_of_2(HW))

    triton_mean_kernel[(BC,)](
        x, out,
        HW,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=min(32, max(1, BLOCK_SIZE // 32)),
    )

    return out


def replacement_func():
    return triton_mean_2d