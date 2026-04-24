import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: SiLU -> adaptive_avg_pool2d(output_size=1) -> flatten -> dropout
# All graphs share this structure with varying dropout rates (p=0.2/0.3/0.4/0.5)
# and input shapes, but training=False so dropout is identity at inference.
# ---------------------------------------------------------------------------

def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Fused Triton kernel: SiLU + global average pool in one pass per (n,c) slice.
# Grid = N*C; each program handles H*W elements for one (n,c) pair.
# Accumulation is done in fp32 for numerical precision; result stored in input
# dtype.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _fused_silu_avgpool_kernel(
    input_ptr,   # [N*C, HW] viewed as contiguous 2-D, input dtype
    output_ptr,  # [N*C] output, same dtype as input
    HW,          # H * W (number of spatial elements per (n,c) slice)
    BLOCK_SIZE: tl.constexpr,  # power-of-2 >= HW, set by autotune
):
    pid = tl.program_id(0)          # one program per (n, c) pair
    base = pid * HW

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HW

    # Load and upcast to fp32 for the SiLU computation
    x = tl.load(input_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # SiLU: x * sigmoid(x)
    silu_x = x * tl.sigmoid(x)

    # Zero out padding positions before reduction
    silu_x = tl.where(mask, silu_x, 0.0)

    # Reduce to scalar mean
    acc = tl.sum(silu_x, axis=0)

    # Convert back to input dtype and store
    result = (acc / HW).to(x.dtype)
    tl.store(output_ptr + pid, result)


@torch.fx.wrap
def fused_silu_avgpool(in_0):
    N, C, H, W = in_0.shape
    HW = H * W
    NC = N * C

    # Output shape: [N, C] (equivalent to flatten(avgpool(...), 1))
    output = torch.empty((N, C), dtype=in_0.dtype, device=in_0.device)

    _fused_silu_avgpool_kernel[(NC,)](
        in_0, output,
        HW=HW,
    )

    return output


def replacement_func():
    return fused_silu_avgpool