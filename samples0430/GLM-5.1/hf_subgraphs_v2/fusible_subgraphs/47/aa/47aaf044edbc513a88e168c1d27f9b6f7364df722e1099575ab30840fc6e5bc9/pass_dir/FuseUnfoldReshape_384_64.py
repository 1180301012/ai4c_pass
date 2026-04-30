import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = in_0.contiguous()
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 384, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 64, 9])
    return (tmp_5,)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def unfold_reshape_kernel_hd64(
    input_ptr, output_ptr,
    stride_b, stride_c, stride_h,
    C, H,
    H_out, num_heads, padding_h,
    HEAD_DIM: tl.constexpr,
    KERNEL_H: tl.constexpr,
):
    row_idx = tl.program_id(0)

    rows_per_batch = H_out * num_heads
    b = row_idx // rows_per_batch
    row_in_batch = row_idx % rows_per_batch
    h_out = row_in_batch // num_heads
    channel_group = row_in_batch % num_heads

    j_range = tl.arange(0, HEAD_DIM)
    c_base = channel_group * HEAD_DIM
    c_offsets = (c_base + j_range) * stride_c

    base_input_offset = b * stride_b

    for k in range(KERNEL_H):
        h_in = h_out + k - padding_h
        valid_h = h_in >= 0 and h_in < H

        input_offsets = base_input_offset + c_offsets + h_in * stride_h
        vals = tl.load(input_ptr + input_offsets, mask=valid_h, other=0.0)

        output_offsets = row_idx * HEAD_DIM * KERNEL_H + j_range * KERNEL_H + k
        tl.store(output_ptr + output_offsets, vals)


@torch.fx.wrap
def unfold_reshape_wrapper_hd64(in_0):
    B, C, H = in_0.shape
    stride_b, stride_c, stride_h = in_0.stride()

    head_dim = 64
    kernel_h = 9
    padding_h = 4
    # H_out = (H + 2*padding_h - dilation*(kernel_h-1) - 1) / stride + 1
    # For padding_h=4, kernel_h=9, stride=1, dilation=1: H_out = (H + 8 - 8 - 1)/1 + 1 = H
    H_out = H
    num_heads = C // head_dim

    total_rows = B * H_out * num_heads

    out = torch.empty((total_rows, head_dim, kernel_h), dtype=in_0.dtype, device=in_0.device)

    unfold_reshape_kernel_hd64[(total_rows,)](
        input_ptr=in_0,
        output_ptr=out,
        stride_b=stride_b, stride_c=stride_c, stride_h=stride_h,
        C=C, H=H,
        H_out=H_out, num_heads=num_heads, padding_h=padding_h,
        HEAD_DIM=head_dim,
        KERNEL_H=kernel_h,
    )

    return out


def replacement_func():
    return unfold_reshape_wrapper_hd64