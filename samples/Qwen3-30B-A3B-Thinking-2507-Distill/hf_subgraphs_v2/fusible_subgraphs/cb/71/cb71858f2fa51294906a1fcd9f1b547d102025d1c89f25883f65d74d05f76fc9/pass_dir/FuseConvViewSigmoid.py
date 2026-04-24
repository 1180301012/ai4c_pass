import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Matches:
        conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_3  = conv2d.view(1, 2, 8, 8)
        tmp_4  = tmp_3.sigmoid()
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(1, 2, 8, 8)
    tmp_4 = tmp_3.sigmoid()
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def _conv_view_sigmoid_kernel(
    in2_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C_IN: tl.constexpr,       # 16
    BLOCK_CIN: tl.constexpr,  # 16
):
    """One program per output channel (128 total).
    C_IN == BLOCK_CIN == 16 so all loads are in-bounds — no masking needed.
    """
    oc = tl.program_id(0)

    cin_offsets = tl.arange(0, BLOCK_CIN)

    # No mask: C_IN == BLOCK_CIN, all accesses are safe
    x = tl.load(in2_ptr + cin_offsets).to(tl.float32)
    w = tl.load(weight_ptr + oc * C_IN + cin_offsets).to(tl.float32)
    bias_val = tl.load(bias_ptr + oc).to(tl.float32)

    dot = tl.sum(x * w, axis=0) + bias_val
    result = tl.sigmoid(dot)

    # out_ptr addresses [1,2,8,8] with the same flat layout as [128]:
    # flat index oc = element [0, oc//8, oc%8, :].
    tl.store(out_ptr + oc, result)


@torch.fx.wrap
def _conv_view_sigmoid_wrapper(in_0, in_1, in_2):
    """
    in_0 : bias   [128]
    in_1 : weight [128, 2, 1, 8]
    in_2 : input  [1,  2, 1, 8]  (CUDA)

    Output allocated as [1,2,8,8] — same flat 128-element layout as the
    kernel's out_ptr, so no extra .view() is needed.
    """
    out = torch.empty((1, 2, 8, 8), dtype=in_2.dtype, device=in_2.device)

    _conv_view_sigmoid_kernel[(128,)](
        in_2, in_1, in_0, out,
        C_IN=16,
        BLOCK_CIN=16,
    )

    return out


def replacement_func():
    return _conv_view_sigmoid_wrapper