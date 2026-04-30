import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.sigmoid(in_0)
    tmp_1 = tmp_0.view(1, 512, 1, 1)
    tmp_2 = in_1 * tmp_1
    tmp_3 = in_1 + tmp_2
    tmp_4 = torch.relu_(tmp_3)
    tmp_5 = torch.nn.functional.dropout2d(tmp_4, 0.1, False, False)
    return (tmp_5,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_se_relu_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    hw_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Determine channel index for each element in the flattened 4D tensor
    # in_1 has shape [1, 512, H, W], so channel = flat_offset // (H * W)
    channel_offsets = offsets // hw_size

    # Load sigmoid input values and cast to float32 for computation
    # tl.sigmoid requires fp32/fp64 input
    sig_vals = tl.load(in_0_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
    sig_results = tl.sigmoid(sig_vals)

    # Load main input values and cast to float32 for computation
    in_1_vals = tl.load(in_1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute: in_1 * (1 + sigmoid(in_0)) then relu, all in float32
    # This is equivalent to: in_1 + in_1 * sigmoid(in_0)
    scale = 1.0 + sig_results
    result = in_1_vals * scale
    result = tl.maximum(result, 0.0)

    # Store - Triton automatically casts float32 to output pointer dtype
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_se_relu(in_0, in_1):
    N = in_1.numel()
    hw_size = in_1.shape[2] * in_1.shape[3]

    out = torch.empty_like(in_1)

    grid = lambda META: ((N + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE'],)

    fused_se_relu_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_elements=N,
        hw_size=hw_size,
    )

    return (out,)


def replacement_func():
    return fused_se_relu