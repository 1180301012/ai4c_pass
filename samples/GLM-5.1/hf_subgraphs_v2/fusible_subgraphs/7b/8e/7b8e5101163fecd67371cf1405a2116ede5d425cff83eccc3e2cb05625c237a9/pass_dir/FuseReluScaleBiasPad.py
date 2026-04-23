import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.ops.aten.relu.default(in_2)
    tmp_3 = torch.ops.aten.mul.Tensor(in_1, tmp_2)
    tmp_4 = torch.ops.aten.add.Tensor(tmp_3, in_0)
    tmp_5 = torch.ops.aten.pad.default(tmp_4, [0, 1, 0, 1], "constant", 0)
    return tmp_5


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_relu_scale_bias_pad_kernel(
    input_ptr, scale_ptr, bias_ptr, output_ptr,
    B, C, H, W,
    stride_ib, stride_ic, stride_ih, stride_iw,
    stride_ob, stride_oc, stride_oh, stride_ow,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    OH = H + 1
    OW = W + 1

    # Convert flat offset to 4D indices for output
    ow = offsets % OW
    oh_temp = offsets // OW
    oh = oh_temp % OH
    oc_temp = oh_temp // OH
    oc = oc_temp % C
    ob = oc_temp // C

    # Check if we're in the valid input region (not padded)
    in_valid_region = (ow < W) & (oh < H)

    # Compute input offsets for valid positions only
    input_offsets = ob * stride_ib + oc * stride_ic + oh * stride_ih + ow * stride_iw
    load_mask = in_valid_region & mask
    input_val = tl.load(input_ptr + input_offsets, mask=load_mask, other=0.0)

    # Load scale and bias (scalar values)
    scale_val = tl.load(scale_ptr)
    bias_val = tl.load(bias_ptr)

    # Compute: scale * relu(input) + bias for valid region, 0 for padded region
    relu_val = tl.where(input_val > 0, input_val, 0.0)
    computed = scale_val * relu_val + bias_val
    output_val = tl.where(in_valid_region, computed, 0.0)

    # Store output
    output_offsets = ob * stride_ob + oc * stride_oc + oh * stride_ow + ow * stride_ow
    tl.store(output_ptr + output_offsets, output_val, mask=mask)


@torch.fx.wrap
def fused_relu_scale_bias_pad(in_0, in_1, in_2):
    B, C, H, W = in_2.shape
    OH = H + 1
    OW = W + 1

    output = torch.empty((B, C, OH, OW), dtype=in_2.dtype, device=in_2.device)

    n_elements = B * C * OH * OW

    def grid(meta):
        return ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_relu_scale_bias_pad_kernel[grid](
        input_ptr=in_2,
        scale_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=output,
        B=B, C=C, H=H, W=W,
        stride_ib=in_2.stride(0), stride_ic=in_2.stride(1),
        stride_ih=in_2.stride(2), stride_iw=in_2.stride(3),
        stride_ob=output.stride(0), stride_oc=output.stride(1),
        stride_oh=output.stride(2), stride_ow=output.stride(3),
        n_elements=n_elements,
    )

    return output


def replacement_func():
    return fused_relu_scale_bias_pad