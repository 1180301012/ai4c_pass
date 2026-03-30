import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: relu(in_1) + in_0  →  .mean([2,3], keepdim=True) method form
# Tries the method-call variant of mean reduction
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.relu(in_1)
    tmp_1 = tmp_0 + in_0
    tmp_2 = tmp_1.mean([2, 3], keepdim=True)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: fused relu + add + global-average-pool
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _fused_relu_add_avgpool_kernel_m(
    in0_ptr,
    in1_ptr,
    out_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    bc = tl.program_id(0)
    base = bc * HW

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    n_full = HW // BLOCK_SIZE
    remainder = HW % BLOCK_SIZE

    for i in range(n_full):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(in0_ptr + base + offsets).to(tl.float32)
        y = tl.load(in1_ptr + base + offsets).to(tl.float32)
        relu_y = tl.where(y > 0.0, y, 0.0)
        acc += relu_y + x

    if remainder > 0:
        offsets = n_full * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW
        x = tl.load(in0_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(in1_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        relu_y = tl.where(y > 0.0, y, 0.0)
        val = relu_y + x
        acc += tl.where(mask, val, 0.0)

    total = tl.sum(acc, axis=0)
    mean_val = total / HW

    tl.store(out_ptr + bc, mean_val)


@torch.fx.wrap
def fused_relu_add_avgpool_m(in_0, in_1):
    in_0 = in_0.contiguous()
    in_1 = in_1.contiguous()

    B, C, H, W = in_0.shape
    HW = H * W
    BC = B * C

    out_f32 = torch.empty(BC, dtype=torch.float32, device=in_0.device)

    _fused_relu_add_avgpool_kernel_m[(BC,)](
        in_0,
        in_1,
        out_f32,
        HW,
    )

    out = out_f32.reshape(B, C, 1, 1).to(in_0.dtype)
    return out


def replacement_func():
    return fused_relu_add_avgpool_m