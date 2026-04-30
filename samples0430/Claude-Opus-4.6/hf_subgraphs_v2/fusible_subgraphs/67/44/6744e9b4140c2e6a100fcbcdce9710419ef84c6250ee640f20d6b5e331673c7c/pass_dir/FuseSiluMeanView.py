import torch
import triton
import triton.language as tl


def pattern(in_1):
    tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_1 = tmp_0.mean((2, 3))
    tmp_4 = tmp_1.view(1, 1, -1)
    return (tmp_0, tmp_4)


def replacement_args(in_1):
    return (in_1,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def silu_mean_kernel(
    input_ptr,
    output_silu_ptr,
    output_mean_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel
    c_idx = tl.program_id(0)
    base_offset = c_idx * HW

    # Accumulate sum for mean in float32
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for start in range(0, HW, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW

        # Load input
        x = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0)

        # Compute SiLU: x * sigmoid(x) in float32 for precision
        x_f32 = x.to(tl.float32)
        sigmoid_x = tl.sigmoid(x_f32)
        silu_x = x_f32 * sigmoid_x

        # Store SiLU output (cast back to input dtype)
        tl.store(output_silu_ptr + base_offset + offsets, silu_x.to(x.dtype), mask=mask)

        # Accumulate for mean
        acc += tl.where(mask, silu_x, 0.0)

    # Compute mean and store
    mean_val = tl.sum(acc, axis=0) / HW
    tl.store(output_mean_ptr + c_idx, mean_val)


@torch.fx.wrap
def fused_silu_mean(in_1):
    B, C, H, W = in_1.shape
    HW = H * W

    # Allocate outputs
    output_silu = torch.empty_like(in_1)
    output_mean = torch.empty((1, 1, C), dtype=in_1.dtype, device=in_1.device)

    # Launch kernel - one program per channel
    grid = (C,)
    silu_mean_kernel[grid](
        in_1, output_silu, output_mean,
        HW,
    )

    return (output_silu, output_mean)


def replacement_func():
    return fused_silu_mean