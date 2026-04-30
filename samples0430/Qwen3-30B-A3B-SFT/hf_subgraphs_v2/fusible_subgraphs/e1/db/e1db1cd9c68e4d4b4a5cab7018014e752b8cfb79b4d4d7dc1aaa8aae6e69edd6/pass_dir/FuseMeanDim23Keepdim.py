import torch
import triton
import triton.language as tl


# ── Pattern: match ONLY mean over dims (2, 3) with keepdim=True (single output)
def pattern(tmp_0):
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_1


def replacement_args(tmp_0):
    return (tmp_0,)


# ── Triton kernel: mean reduction over last two dims ──────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8,  num_stages=3),
    ],
    key=['HW'],
)
@triton.jit
def triton_mean_dim23_kernel(
    in_ptr,
    out_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (batch, channel) pair
    pid = tl.program_id(0)
    base = pid * HW
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW

    x = tl.load(in_ptr + base + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Reduce sum and divide
    total = tl.sum(x_f32, axis=0)
    mean_val = total / HW

    # Store back in original dtype
    tl.store(out_ptr + pid, mean_val.to(x.dtype))


# ── Kernel wrapper ────────────────────────────────────────────────────────────
@torch.fx.wrap
def triton_mean_dim23(tmp_0):
    B, C, H, W = tmp_0.shape
    HW = H * W
    BC = B * C

    # Output shape: [B, C, 1, 1]
    out = torch.empty((B, C, 1, 1), dtype=tmp_0.dtype, device=tmp_0.device)

    triton_mean_dim23_kernel[(BC,)](tmp_0, out, HW)
    return out


def replacement_func():
    return triton_mean_dim23