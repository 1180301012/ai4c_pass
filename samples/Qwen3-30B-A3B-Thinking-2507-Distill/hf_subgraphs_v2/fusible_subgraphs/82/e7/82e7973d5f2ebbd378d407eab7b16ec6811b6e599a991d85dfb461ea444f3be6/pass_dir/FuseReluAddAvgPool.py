import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: ReLU(in_1) + in_0  then  AdaptiveAvgPool2d(result, 1)
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
    tmp_1 = tmp_0 + in_0
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Fused Triton kernel
#   Grid: (B * C,)  — one program per (batch, channel) slice
#   Each program:
#     1. Loads BLOCK_HW spatial elements from in_0 and in_1
#     2. Applies ReLU to in_1 elements
#     3. Adds them together
#     4. Accumulates in fp32, divides by HW → mean
#     5. Stores scalar result back in the original dtype
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256}, num_warps=1),
        triton.Config({'BLOCK_HW': 256}, num_warps=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
    ],
    key=['HW', 'dtype'],
)
@triton.jit
def fused_relu_add_avgpool_kernel(
    in0_ptr, in1_ptr, out_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    # One program per (batch, channel) pair
    bc = tl.program_id(0)
    base = bc * HW

    offsets = tl.arange(0, BLOCK_HW)
    mask = offsets < HW

    # Load inputs — remain in original dtype for as long as possible
    x = tl.load(in0_ptr + base + offsets, mask=mask, other=0.0)
    y = tl.load(in1_ptr + base + offsets, mask=mask, other=0.0)

    # Upcast to float32 for accurate accumulation
    x_f = x.to(tl.float32)
    y_f = y.to(tl.float32)

    # relu(in_1) + in_0
    y_f = tl.where(y_f > 0.0, y_f, 0.0)
    z_f = x_f + y_f

    # Sum-reduce then divide to get the mean
    total = tl.sum(z_f, axis=0)
    avg = total / HW  # already float32

    # Store — Triton auto-truncates fp32 → fp16/bf16 when pointer type requires it
    tl.store(out_ptr + bc, avg)


@torch.fx.wrap
def fused_relu_add_avgpool(in_0, in_1):
    B, C, H, W = in_0.shape
    HW = H * W

    # Output shape matches adaptive_avg_pool2d output_size=1: [B, C, 1, 1]
    out = torch.empty((B, C, 1, 1), dtype=in_0.dtype, device=in_0.device)

    fused_relu_add_avgpool_kernel[(B * C,)](
        in_0, in_1, out,
        HW=HW,
        dtype=in_0.dtype,
    )

    return out


def replacement_func():
    return fused_relu_add_avgpool