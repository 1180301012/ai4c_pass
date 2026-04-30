import torch
import triton
import triton.language as tl


def pattern(conv_output, in_2):
    tmp_3 = torch.sigmoid(conv_output)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    tmp_5 = in_2 * tmp_4
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(conv_output, in_2):
    return (conv_output, in_2)


@triton.jit
def fused_sigmoid_view_mul_kernel(
    in_2_ptr,
    conv_out_ptr,
    out_ptr,
    n_elements,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Determine channel index for each element
    # For a [1, C, H, W] tensor: flat_offset = c * HW + spatial_idx
    channel_idx = offsets // HW

    # Load sigmoid input and compute sigmoid (cast to fp32 for bf16/fp16 support)
    sigmoid_in = tl.load(conv_out_ptr + channel_idx, mask=mask, other=0.0)
    sigmoid_val = tl.sigmoid(sigmoid_in.to(tl.float32))

    # Load in_2 value
    in_2_val = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)

    # Multiply
    out_val = sigmoid_val * in_2_val

    # Store
    tl.store(out_ptr + offsets, out_val, mask=mask)


@torch.fx.wrap
def fused_sigmoid_view_mul(conv_output, in_2):
    C = conv_output.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]
    n_elements = in_2.numel()
    HW = H * W

    out = torch.empty_like(in_2)

    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_sigmoid_view_mul_kernel[(num_programs,)](
        in_2_ptr=in_2,
        conv_out_ptr=conv_output,
        out_ptr=out,
        n_elements=n_elements,
        HW=HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_sigmoid_view_mul