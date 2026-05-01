import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------
# Pattern: ReLU(inplace=True) followed by mean over spatial dims (2,3)
# Both the ReLU output AND the mean are returned (both are observable).
# -----------------------------------------------------------------------
def pattern(x):
    tmp_0 = torch.nn.functional.relu(x, inplace=True)
    tmp_3 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_3)


def replacement_args(x):
    return (x,)


# -----------------------------------------------------------------------
# Fused Triton kernel: one program per (batch, channel) slice.
# Each program loads HW elements, applies ReLU, stores back, and
# accumulates a float32 sum for the mean.
# -----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def _relu_spatial_mean_kernel(
    x_ptr,
    out_relu_ptr,
    out_mean_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    # One program handles a single (batch, channel) slice of length HW
    pid   = tl.program_id(0)
    base  = pid * HW

    # Float32 accumulator for the mean (avoids overflow in fp16/bf16)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for block_start in range(0, HW, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask    = offsets < HW

        # Load (OOB elements are 0.0, relu(0)=0 → safe to accumulate)
        x       = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)

        # ReLU in original dtype
        x_relu  = tl.where(x > 0, x, 0.0)

        # Store ReLU result
        tl.store(out_relu_ptr + base + offsets, x_relu, mask=mask)

        # Accumulate in fp32 for precision
        acc += x_relu.to(tl.float32)

    # Compute mean and store to the correct output dtype
    total    = tl.sum(acc, axis=0)
    mean_val = total / HW
    tl.store(out_mean_ptr + pid, mean_val.to(out_mean_ptr.dtype.element_ty))


# -----------------------------------------------------------------------
# Host wrapper — must be decorated with @torch.fx.wrap
# -----------------------------------------------------------------------
@torch.fx.wrap
def relu_spatial_mean_fused(x):
    B, C, H, W = x.shape
    HW = H * W

    # Outputs
    out_relu = torch.empty_like(x)
    # Shape (B, C, 1, 1) is contiguous; element [b,c,0,0] is at offset b*C+c
    # which matches program_id = b*C + c exactly.
    out_mean = torch.empty((B, C, 1, 1), dtype=x.dtype, device=x.device)

    _relu_spatial_mean_kernel[(B * C,)](
        x, out_relu, out_mean, HW,
    )

    return (out_relu, out_mean)


def replacement_func():
    return relu_spatial_mean_fused