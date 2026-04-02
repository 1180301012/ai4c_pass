import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(conv2d, 2)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        # 8 focused configs – best balance found empirically
        triton.Config({'BLOCK_HW': 128,  'BLOCK_CI': 32, 'BLOCK_CO': 32}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_HW': 256,  'BLOCK_CI': 32, 'BLOCK_CO': 32}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_HW': 512,  'BLOCK_CI': 32, 'BLOCK_CO': 32}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_HW': 1024, 'BLOCK_CI': 32, 'BLOCK_CO': 32}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_HW': 128,  'BLOCK_CI': 32, 'BLOCK_CO': 32}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_HW': 256,  'BLOCK_CI': 32, 'BLOCK_CO': 32}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_HW': 256,  'BLOCK_CI': 64, 'BLOCK_CO': 32}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_HW': 512,  'BLOCK_CI': 64, 'BLOCK_CO': 32}, num_warps=8,  num_stages=3),
    ],
    key=['N', 'C_in', 'C_out', 'HW'],
)
@triton.jit
def fused_conv1x1_flatten_kernel(
    input_ptr,   # [N, C_in, HW] contiguous
    weight_ptr,  # [C_out, C_in] contiguous (weight viewed as 2D)
    bias_ptr,    # [C_out]
    output_ptr,  # [N, C_out, HW]
    N, C_in, C_out, HW,
    BLOCK_HW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """
    Fused 1x1 conv2d + flatten kernel.
    Grid: (N, ceil(HW/BLOCK_HW))
    """
    pid_n  = tl.program_id(0)
    pid_hw = tl.program_id(1)

    hw_start   = pid_hw * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask    = hw_offsets < HW

    co_offsets = tl.arange(0, BLOCK_CO)
    co_mask    = co_offsets < C_out

    acc = tl.zeros((BLOCK_CO, BLOCK_HW), dtype=tl.float32)

    for ci_start in range(0, C_in, BLOCK_CI):
        ci_offsets = ci_start + tl.arange(0, BLOCK_CI)
        ci_mask    = ci_offsets < C_in

        # Load weight tile [BLOCK_CO, BLOCK_CI]
        w = tl.load(
            weight_ptr + co_offsets[:, None] * C_in + ci_offsets[None, :],
            mask=co_mask[:, None] & ci_mask[None, :],
            other=0.0
        ).to(tl.float32)

        # Load input tile [BLOCK_CI, BLOCK_HW]
        x = tl.load(
            input_ptr + pid_n * (C_in * HW) + ci_offsets[:, None] * HW + hw_offsets[None, :],
            mask=ci_mask[:, None] & hw_mask[None, :],
            other=0.0
        ).to(tl.float32)

        acc = tl.dot(w, x, acc)

    # Add bias [BLOCK_CO]
    b = tl.load(bias_ptr + co_offsets, mask=co_mask, other=0.0).to(tl.float32)
    acc += b[:, None]

    # Store output [BLOCK_CO, BLOCK_HW]
    out_ptrs = output_ptr + pid_n * (C_out * HW) + co_offsets[:, None] * HW + hw_offsets[None, :]
    tl.store(out_ptrs, acc, mask=co_mask[:, None] & hw_mask[None, :])


@torch.fx.wrap
def fused_conv1x1_flatten(bias, weight, input_tensor):
    """
    Replacement for: conv2d(input, weight, bias, stride=1, pad=0, dil=1, groups=1) + flatten(2)
    where weight has kernel size 1x1.

    Args:
        bias:         [C_out]           bias tensor
        weight:       [C_out, C_in, 1, 1] weight tensor
        input_tensor: [N, C_in, H, W]  input feature map

    Returns:
        Tuple with single tensor of shape [N, C_out, H*W]
    """
    # Ensure contiguous memory layout
    input_tensor = input_tensor.contiguous()

    N, C_in, H, W = input_tensor.shape
    C_out = weight.shape[0]
    HW    = H * W

    # For small N the Triton kernel has insufficient parallelism to saturate
    # the GPU.  The @ operator (batched matmul) handles all small-N cases via
    # cuBLAS, which has specialised algorithms for small batch sizes.
    #   weight[C_out,C_in] @ input[N,C_in,HW]  →  output[N,C_out,HW]
    # The 2-D × 3-D matmul broadcasts weight over the batch dimension N.
    _TRITON_N_THRESHOLD = 16
    if N <= _TRITON_N_THRESHOLD:
        w   = weight.view(C_out, C_in)
        x   = input_tensor.view(N, C_in, HW)
        out = w @ x                          # [N, C_out, HW]
        out = out + bias.view(C_out, 1)      # broadcast bias [C_out,1]
        return out

    # For larger batches use the fused Triton kernel.
    bias_cont = bias.contiguous()
    weight_2d = weight.view(C_out, C_in).contiguous()

    # Allocate output directly in the flattened shape [N, C_out, HW]
    output = torch.empty(
        (N, C_out, HW),
        dtype=input_tensor.dtype,
        device=input_tensor.device
    )

    # Dynamic grid: (batch, hw_tiles) – autotune selects best BLOCK_HW per shape
    grid = lambda meta: (N, triton.cdiv(HW, meta['BLOCK_HW']))

    fused_conv1x1_flatten_kernel[grid](
        input_tensor, weight_2d, bias_cont, output,
        N, C_in, C_out, HW,
    )

    return output


def replacement_func():
    return fused_conv1x1_flatten