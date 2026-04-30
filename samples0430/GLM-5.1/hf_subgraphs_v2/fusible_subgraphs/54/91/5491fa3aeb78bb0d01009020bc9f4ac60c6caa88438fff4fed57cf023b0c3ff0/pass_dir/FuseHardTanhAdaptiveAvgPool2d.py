import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        # Small BLOCK_NC for maximum parallelism (good for small total_nc)
        triton.Config({'BLOCK_NC': 1, 'BLOCK_HW': 32}),
        triton.Config({'BLOCK_NC': 1, 'BLOCK_HW': 64}),
        triton.Config({'BLOCK_NC': 1, 'BLOCK_HW': 128}),
        triton.Config({'BLOCK_NC': 1, 'BLOCK_HW': 256}),
        # Moderate BLOCK_NC for better amortization
        triton.Config({'BLOCK_NC': 4, 'BLOCK_HW': 64}),
        triton.Config({'BLOCK_NC': 8, 'BLOCK_HW': 64}),
        # Larger BLOCK_NC for large total_nc
        triton.Config({'BLOCK_NC': 16, 'BLOCK_HW': 64}),
        triton.Config({'BLOCK_NC': 32, 'BLOCK_HW': 64}),
    ],
    key=['total_nc', 'total_hw'],
)
@triton.jit
def fused_hardtanh_avgpool_kernel(
    input_ptr, output_ptr,
    total_nc, total_hw,
    BLOCK_NC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    nc_start = pid * BLOCK_NC

    nc_offsets = nc_start + tl.arange(0, BLOCK_NC)
    nc_mask = nc_offsets < total_nc

    # Accumulator in float32 for numerical accuracy
    acc = tl.zeros([BLOCK_NC], dtype=tl.float32)

    # Iterate over spatial dimensions in blocks
    for hw_start in range(0, total_hw, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < total_hw

        # 2D indexing: input[nc, hw] maps to flat layout [N*C, H*W]
        input_offsets = nc_offsets[:, None] * total_hw + hw_offsets[None, :]
        mask_2d = nc_mask[:, None] & hw_mask[None, :]

        # Load input values and convert to float32 for computation
        values = tl.load(input_ptr + input_offsets, mask=mask_2d, other=0.0)
        values_f32 = values.to(tl.float32)

        # Apply hardtanh: clamp between 0 and 6
        clamped = tl.minimum(tl.maximum(values_f32, 0.0), 6.0)

        # Accumulate partial sums over HW dimension
        acc += tl.sum(clamped, axis=1)

    # Divide by total_hw to compute average (adaptive_avg_pool2d)
    avg = acc / total_hw

    # Store output - Triton auto-converts float32 to output dtype
    tl.store(output_ptr + nc_offsets, avg, mask=nc_mask)


@torch.fx.wrap
def fused_hardtanh_avgpool(in_0):
    N, C, H, W = in_0.shape
    total_nc = N * C
    total_hw = H * W

    # Output shape matches adaptive_avg_pool2d output: [N, C, 1, 1]
    out = torch.empty((N, C, 1, 1), dtype=in_0.dtype, device=in_0.device)

    # Grid size depends on autotuned BLOCK_NC
    grid = lambda meta: ((total_nc + meta['BLOCK_NC'] - 1) // meta['BLOCK_NC'],)

    fused_hardtanh_avgpool_kernel[grid](
        input_ptr=in_0,
        output_ptr=out,
        total_nc=total_nc,
        total_hw=total_hw,
    )

    return out


def replacement_func():
    return fused_hardtanh_avgpool