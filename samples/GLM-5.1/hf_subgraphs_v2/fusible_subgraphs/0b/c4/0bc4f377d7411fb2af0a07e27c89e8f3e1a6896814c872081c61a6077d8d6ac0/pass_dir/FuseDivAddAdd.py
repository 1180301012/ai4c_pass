import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_0 = in_0 / 8.0
    tmp_0 = tmp_0 + in_2
    tmp_2 = tmp_0 + in_1
    return (tmp_2,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_div_add_add_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, out_ptr,
    n_elements,
    stride_in1_0, stride_in1_1, stride_in1_2, stride_in1_3,
    dim_H, dim_W, dim_D,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load in_0 and in_2 (same shape as output, contiguous)
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)

    # Compute 4D index from flat offset for in_1 broadcast
    # Output shape: [B, H, W, D] = [2, 12, 7, 7]
    # in_1 shape: [2, 1, 1, 7] -> broadcasts along dims 1 and 2
    d = offsets % dim_D
    w = (offsets // dim_D) % dim_W
    h = (offsets // (dim_W * dim_D)) % dim_H
    b = offsets // (dim_H * dim_W * dim_D)

    in1_offset = b * stride_in1_0 + h * stride_in1_1 + w * stride_in1_2 + d * stride_in1_3
    in_1 = tl.load(in_1_ptr + in1_offset, mask=mask, other=0.0)

    # Compute: out = (in_0 / 8.0) + in_2 + in_1
    out = in_0 / 8.0 + in_2 + in_1

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_div_add_add(in_0, in_1, in_2):
    N = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(in_0)

    # Compute broadcast strides for in_1
    # in_1 has shape [2, 1, 1, 7], needs to broadcast to [2, 12, 7, 7]
    # For broadcast dims (where in_1 size = 1 and output size > 1), stride = 0
    in1_orig_stride = in_1.stride()
    in1_shape = in_1.shape
    out_shape = in_0.shape
    in1_strides = []
    for i in range(4):
        if in1_shape[i] == 1 and out_shape[i] > 1:
            in1_strides.append(0)
        else:
            in1_strides.append(in1_orig_stride[i])

    # Shape dimensions for decomposing flat index
    dim_D = out_shape[3]
    dim_W = out_shape[2]
    dim_H = out_shape[1]

    fused_div_add_add_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_ptr=out,
        n_elements=N,
        stride_in1_0=in1_strides[0],
        stride_in1_1=in1_strides[1],
        stride_in1_2=in1_strides[2],
        stride_in1_3=in1_strides[3],
        dim_H=dim_H,
        dim_W=dim_W,
        dim_D=dim_D,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_div_add_add