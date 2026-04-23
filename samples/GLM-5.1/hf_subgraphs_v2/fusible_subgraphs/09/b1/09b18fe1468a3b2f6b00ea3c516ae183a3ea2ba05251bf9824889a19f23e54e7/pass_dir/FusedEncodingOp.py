import torch
import triton
import triton.language as tl


def pattern(in_0, in_4):
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    tmp_10 = tmp_8 - tmp_6
    return tmp_10


def replacement_args(in_0, in_4):
    return (in_0, in_4)


@triton.jit
def fused_broadcast_sub_kernel(
    in_0_ptr, in_4_ptr, out_ptr,
    stride_in0_k, stride_in0_f,
    stride_in4_p, stride_in4_f,
    stride_out_p, stride_out_k, stride_out_f,
    num_codewords: tl.constexpr, num_features: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    p = tl.program_id(0)
    f_block = tl.program_id(1)

    f_start = f_block * BLOCK_F
    f_offsets = f_start + tl.arange(0, BLOCK_F)
    f_mask = f_offsets < num_features

    in_4_vals = tl.load(in_4_ptr + p * stride_in4_p + f_offsets * stride_in4_f, mask=f_mask, other=0.0).to(tl.float32)

    k_offsets = tl.arange(0, num_codewords)

    in_0_offsets = k_offsets[:, None] * stride_in0_k + f_offsets[None, :] * stride_in0_f
    in_0_vals = tl.load(in_0_ptr + in_0_offsets, mask=f_mask[None, :], other=0.0).to(tl.float32)

    out_vals = in_4_vals[None, :] - in_0_vals

    out_offsets = p * stride_out_p + k_offsets[:, None] * stride_out_k + f_offsets[None, :] * stride_out_f
    tl.store(out_ptr + out_offsets, out_vals.to(out_ptr.dtype.element_ty), mask=f_mask[None, :])


@torch.fx.wrap
def kernel_wrapper(in_0, in_4):
    num_pixels = in_4.shape[1]
    num_codewords = 32
    num_features = in_4.shape[2]

    out = torch.empty((1, num_pixels, num_codewords, num_features), dtype=in_0.dtype, device=in_0.device)

    BLOCK_F = 256
    num_f_blocks = (num_features + BLOCK_F - 1) // BLOCK_F
    fused_broadcast_sub_kernel[(num_pixels, num_f_blocks)](
        in_0_ptr=in_0, in_4_ptr=in_4, out_ptr=out,
        stride_in0_k=in_0.stride(0), stride_in0_f=in_0.stride(1),
        stride_in4_p=in_4.stride(1), stride_in4_f=in_4.stride(2),
        stride_out_p=out.stride(1), stride_out_k=out.stride(2), stride_out_f=out.stride(3),
        num_codewords=num_codewords, num_features=num_features,
        BLOCK_F=BLOCK_F,
    )

    return out


def replacement_func():
    return kernel_wrapper