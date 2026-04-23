import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
    tmp_1 = tmp_0 + in_0
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return (tmp_2,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32}, num_warps=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=2),
        triton.Config({'BLOCK_HW': 32}, num_warps=4),
        triton.Config({'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 64}, num_warps=8),
        triton.Config({'BLOCK_HW': 128}, num_warps=8),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def fused_relu_add_avg_pool_kernel(
    in0_ptr, in1_ptr, out_ptr,
    C, HW,
    BLOCK_HW: tl.constexpr,
    DTYPE: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)

    # Offset into input tensors for this (n, c) pair
    # Input layout: [N, C, H, W] -> all H*W elements for (n,c) are contiguous
    nc_offset = n * C * HW + c * HW

    # Accumulate in float32 for numerical accuracy
    acc = 0.0

    for hw_start in range(0, HW, BLOCK_HW):
        offsets = hw_start + tl.arange(0, BLOCK_HW)
        mask = offsets < HW

        # Load input elements and upcast to float32
        in0_val = tl.load(in0_ptr + nc_offset + offsets, mask=mask, other=0.0).to(tl.float32)
        in1_val = tl.load(in1_ptr + nc_offset + offsets, mask=mask, other=0.0).to(tl.float32)

        # ReLU(in1) + in0
        relu_val = tl.maximum(in1_val, 0.0)
        val = relu_val + in0_val

        # Accumulate only valid elements
        acc += tl.where(mask, val, 0.0)

    # Divide by H*W to get the average
    result = acc / HW

    # Store to output at [n, c, 0, 0]
    # Output layout: [N, C, 1, 1] -> contiguous, out[n,c,0,0] at offset n*C+c
    out_offset = n * C + c
    tl.store(out_ptr + out_offset, result.to(DTYPE))


@torch.fx.wrap
def fused_relu_add_avg_pool(in_0, in_1):
    N, C, H, W = in_0.shape
    HW = H * W
    dtype = in_0.dtype

    # Map torch dtype to Triton dtype for constexpr
    dtype_map = {
        torch.float16: tl.float16,
        torch.float32: tl.float32,
        torch.bfloat16: tl.bfloat16,
    }
    triton_dtype = dtype_map[dtype]

    # Create output tensor with correct shape and dtype
    out = torch.empty((N, C, 1, 1), dtype=dtype, device=in_0.device)

    # 2D grid: each program handles one (batch, channel) output element
    grid = (N, C)

    fused_relu_add_avg_pool_kernel[grid](
        in0_ptr=in_0, in1_ptr=in_1, out_ptr=out,
        C=C, HW=HW,
        DTYPE=triton_dtype,
    )

    return out


def replacement_func():
    return fused_relu_add_avg_pool