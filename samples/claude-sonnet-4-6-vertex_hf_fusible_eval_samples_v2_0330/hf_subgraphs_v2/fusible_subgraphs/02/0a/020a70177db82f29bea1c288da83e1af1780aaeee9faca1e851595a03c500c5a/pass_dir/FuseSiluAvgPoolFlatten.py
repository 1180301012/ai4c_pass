import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def fused_silu_avgpool_kernel(
    input_ptr,
    output_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    bc_idx = tl.program_id(0)
    input_base = bc_idx * HW

    # Accumulate SiLU values over H*W spatial positions
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for start in range(0, HW, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW
        # Load input element (out-of-bounds -> 0.0, and silu(0) = 0)
        x = tl.load(input_ptr + input_base + offsets, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        sigmoid_x = 1.0 / (1.0 + tl.exp(-x_f32))
        silu_x = x_f32 * sigmoid_x
        acc += silu_x  # masked elements have x=0 -> silu(0)=0, so safe

    # Compute mean and write output
    result = tl.sum(acc) / HW
    tl.store(output_ptr + bc_idx, result)


@torch.fx.wrap
def fused_silu_avgpool_flatten(in_0):
    B, C, H, W = in_0.shape
    HW = H * W
    BC = B * C

    # Accumulate in float32 for numerical accuracy, then cast back
    out = torch.empty((BC,), dtype=torch.float32, device=in_0.device)

    fused_silu_avgpool_kernel[(BC,)](
        in_0,
        out,
        HW,
    )

    # Reshape to [B, C] and cast to original dtype
    return out.reshape(B, C).to(in_0.dtype)


def replacement_func():
    return fused_silu_avgpool_flatten