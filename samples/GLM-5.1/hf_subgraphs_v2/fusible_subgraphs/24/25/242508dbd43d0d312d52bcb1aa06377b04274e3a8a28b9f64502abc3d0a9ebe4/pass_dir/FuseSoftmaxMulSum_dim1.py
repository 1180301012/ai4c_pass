import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return (tmp_2,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_softmax_mul_sum_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    dim1_stride_in_0,
    dim1_stride_in_1,
    channel_stride_in_0,
    channel_stride_in_1,
    channel_stride_out,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    c = tl.program_id(0)
    spatial_pid = tl.program_id(1)
    off = spatial_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < spatial_size

    # Base offsets for this channel for slice 0 and slice 1
    base0 = c * channel_stride_in_0
    base1 = base0 + dim1_stride_in_0
    in_1_base = c * channel_stride_in_1
    out_base = c * channel_stride_out

    # Load softmax input values (scalar per channel)
    x0 = tl.load(in_1_ptr + in_1_base).to(tl.float32)
    x1 = tl.load(in_1_ptr + in_1_base + dim1_stride_in_1).to(tl.float32)

    # Numerically stable softmax for 2 elements
    mx = tl.maximum(x0, x1)
    e0 = tl.exp(x0 - mx)
    e1 = tl.exp(x1 - mx)
    s = e1 / (e0 + e1)  # softmax weight for slice 1
    # result = v0*(1-s) + v1*s = v0 + s*(v1-v0)

    # Load in_0 for both slices
    v0 = tl.load(in_0_ptr + base0 + off, mask=mask, other=0.0).to(tl.float32)
    v1 = tl.load(in_0_ptr + base1 + off, mask=mask, other=0.0).to(tl.float32)

    # Fused compute
    out = v0 + s * (v1 - v0)

    tl.store(out_ptr + out_base + off, out, mask=mask)


@torch.fx.wrap
def fused_softmax_mul_sum(in_0, in_1):
    out_shape = list(in_0.shape)
    out_shape.pop(1)
    out = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)

    n_channels = in_0.shape[2]
    spatial_size = 1
    for d in range(3, len(in_0.shape)):
        spatial_size *= in_0.shape[d]

    # Strides
    dim1_stride_in_0 = in_0.stride(1)
    dim1_stride_in_1 = in_1.stride(1)
    channel_stride_in_0 = in_0.stride(2)
    channel_stride_in_1 = in_1.stride(2)
    channel_stride_out = out.stride(1)

    BLOCK_SIZE = 512
    n_tiles = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_softmax_mul_sum_kernel[(n_channels, n_tiles)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        dim1_stride_in_0=dim1_stride_in_0,
        dim1_stride_in_1=dim1_stride_in_1,
        channel_stride_in_0=channel_stride_in_0,
        channel_stride_in_1=channel_stride_in_1,
        channel_stride_out=channel_stride_out,
        spatial_size=spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_softmax_mul_sum