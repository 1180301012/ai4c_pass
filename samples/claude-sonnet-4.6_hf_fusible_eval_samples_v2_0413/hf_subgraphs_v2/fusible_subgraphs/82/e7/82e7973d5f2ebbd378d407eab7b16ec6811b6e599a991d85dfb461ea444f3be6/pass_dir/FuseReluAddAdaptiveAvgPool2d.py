import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: relu(in_1) + in_0  followed by global average-pool (output_size=1)
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
    tmp_1 = tmp_0 + in_0
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: one program per (b, c) pair.
# Each program streams over the H*W spatial elements, applies relu+add, and
# accumulates into a float32 register vector before writing the mean.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},  num_warps=1),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _fused_relu_add_avgpool_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    BC,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid   = tl.program_id(0)          # flat index over B*C
    base  = pid * HW                  # start of this channel's spatial block

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Stream over all HW elements in chunks of BLOCK_SIZE
    for start in range(0, HW, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask    = offsets < HW

        x0 = tl.load(in0_ptr + base + offsets, mask=mask, other=0.0)
        x1 = tl.load(in1_ptr + base + offsets, mask=mask, other=0.0)

        # Upcast to float32 for accurate accumulation
        x0_f = x0.to(tl.float32)
        x1_f = x1.to(tl.float32)

        relu_x1 = tl.maximum(x1_f, 0.0)
        result  = relu_x1 + x0_f
        result  = tl.where(mask, result, 0.0)

        acc = acc + result

    # Reduce to scalar mean, then store (Triton auto-casts to output dtype)
    total    = tl.sum(acc)
    mean_val = total / HW

    tl.store(out_ptr + pid, mean_val)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_relu_add_avgpool(in_0, in_1):
    B, C, H, W = in_0.shape
    HW = H * W
    BC = B * C

    # Output: [B, C, 1, 1], same dtype as inputs
    out = torch.empty((B, C, 1, 1), dtype=in_0.dtype, device=in_0.device)

    _fused_relu_add_avgpool_kernel[(BC,)](
        in_0, in_1, out,
        BC, HW,
    )
    return out


# ---------------------------------------------------------------------------
# replacement_func must be zero-arg and return the callable (not call it)
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_relu_add_avgpool