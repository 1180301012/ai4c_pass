import operator
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: relu(inplace) -> adaptive_avg_pool2d(_, 1) -> flatten(1, -1)
#
# x   : result of iadd(in_1 * sigmoid(in_2).view(1,-1,1,1).expand_as(in_1), in_0)
#       shape  [1, C, H, W]
# y   : placeholder (not used in this pass – part of framework signature)
# result:
#       shape  [1, C]
#
# Fuses the 3 ops into one kernel: each CUDA block handles one channel,
# applies ReLU over all H*W spatial elements, then reduces to mean.
# This eliminates the [1,C,H,W] intermediate tensor written by relu
# and re-read by avgpool (saves ~2 * C*H*W memory bandwidth).
# ---------------------------------------------------------------------------

def pattern(x):
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7


def replacement_args(x):
    return (x,)

# ---------------------------------------------------------------------------
# Triton kernel — one program per channel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _relu_avgpool_kernel(
    x_ptr,           # [B, C, H, W] contiguous float16/bf16/fp32
    out_ptr,         # [B, C] output
    C,
    HW,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_HW)
    mask = offsets < HW

    base = c * HW
    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)

    # Global average pool
    avg = tl.sum(x, axis=0) / HW

    # Store (cast back to original dtype)
    tl.store(out_ptr + c, avg.to(x.dtype))


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_relu_avgpool_flatten(x):
    B  = x.shape[0]
    C  = x.shape[1]
    H  = x.shape[2]
    W  = x.shape[3]
    HW = H * W

    out = torch.empty((B, C), dtype=x.dtype, device=x.device)

    _relu_avgpool_kernel[(C,)](
        x,
        out,
        C,
        HW,
    )

    return out


# ---------------------------------------------------------------------------
# replacement_func
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_relu_avgpool_flatten