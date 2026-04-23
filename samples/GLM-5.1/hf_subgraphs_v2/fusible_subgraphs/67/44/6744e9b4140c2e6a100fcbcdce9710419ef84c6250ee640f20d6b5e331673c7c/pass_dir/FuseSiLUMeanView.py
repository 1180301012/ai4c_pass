import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.silu(in_1)
    tmp_1 = tmp_0.mean((2, 3))
    tmp_4 = tmp_1.view(1, 1, -1)
    return (tmp_0, tmp_4)


def replacement_args(in_0, in_1):
    return (in_1,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64}, num_warps=2),
        triton.Config({'BLOCK_M': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 256}, num_warps=4),
        triton.Config({'BLOCK_M': 512}, num_warps=8),
        triton.Config({'BLOCK_M': 1024}, num_warps=16),
    ],
    key=['C', 'HW'],
)
@triton.jit
def silu_mean_kernel(
    input_ptr,
    silu_output_ptr,
    sum_ptr,
    C,
    HW,
    BLOCK_M: tl.constexpr,
):
    channel = tl.program_id(0)
    block_idx = tl.program_id(1)

    spatial_start = block_idx * BLOCK_M
    spatial_offsets = spatial_start + tl.arange(0, BLOCK_M)
    spatial_mask = spatial_offsets < HW

    offsets = channel * HW + spatial_offsets

    # Load and compute SiLU in float32 for numerical stability
    x = tl.load(input_ptr + offsets, mask=spatial_mask, other=0.0).to(tl.float32)
    silu_val = x * tl.sigmoid(x)

    # Store SiLU output (Triton casts float32 back to ptr dtype automatically)
    tl.store(silu_output_ptr + offsets, silu_val, mask=spatial_mask)

    # Accumulate partial sum in float32 for mean computation
    # masked-out elements have x=0, silu(0)=0, so they contribute 0 to sum
    partial_sum = tl.sum(silu_val)
    tl.atomic_add(sum_ptr + channel, partial_sum)


@triton.jit
def mean_finalize_kernel(
    sum_ptr,
    mean_output_ptr,
    C,
    HW,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    channel_start = pid * BLOCK_C
    channel_offsets = channel_start + tl.arange(0, BLOCK_C)
    mask = channel_offsets < C

    sum_val = tl.load(sum_ptr + channel_offsets, mask=mask)
    mean_val = sum_val / HW

    # Store to mean_output (shape [1,1,C] contiguous, offset=channel works)
    # Triton casts float32 to ptr dtype automatically
    tl.store(mean_output_ptr + channel_offsets, mean_val, mask=mask)


@torch.fx.wrap
def silu_mean_fused(x):
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    HW = H * W

    silu_output = torch.empty_like(x)
    sum_buffer = torch.zeros(C, dtype=torch.float32, device=x.device)

    # Dynamic grid based on autotuned BLOCK_M
    grid = lambda meta: (C, triton.cdiv(HW, meta['BLOCK_M']))

    silu_mean_kernel[grid](
        input_ptr=x,
        silu_output_ptr=silu_output,
        sum_ptr=sum_buffer,
        C=C,
        HW=HW,
    )

    # Create mean output directly in [1, 1, C] shape (avoids blocked torch.view)
    mean_output = torch.empty((1, 1, C), dtype=x.dtype, device=x.device)

    BLOCK_C = 512
    grid_finalize = (triton.cdiv(C, BLOCK_C),)

    mean_finalize_kernel[grid_finalize](
        sum_ptr=sum_buffer,
        mean_output_ptr=mean_output,
        C=C,
        HW=HW,
        BLOCK_C=BLOCK_C,
    )

    return (silu_output, mean_output)


def replacement_func():
    return silu_mean_fused