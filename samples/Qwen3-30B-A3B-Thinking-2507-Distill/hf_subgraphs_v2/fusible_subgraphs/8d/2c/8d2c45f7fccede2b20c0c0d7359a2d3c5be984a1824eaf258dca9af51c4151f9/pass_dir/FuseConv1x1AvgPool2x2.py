import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.avg_pool2d(tmp_1, 2, 2, 0, False, True, None)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 16, 'BLOCK_NHW': 16}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_NHW': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_NHW': 16}, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_NHW': 32}, num_warps=8),
        triton.Config({'BLOCK_C': 64, 'BLOCK_NHW': 16}, num_warps=8),
        triton.Config({'BLOCK_C': 64, 'BLOCK_NHW': 32}, num_warps=8),
    ],
    key=['C_IN', 'H_IN', 'W_IN', 'C_OUT'],
)
@triton.jit
def _fused_conv1x1_avgpool_kernel(
    input_ptr,   # [N, C_IN, H_IN, W_IN] contiguous NCHW
    weight_ptr,  # [C_OUT, C_IN, 1, 1]    (effectively [C_OUT, C_IN])
    output_ptr,  # [N, C_OUT, H_OUT, W_OUT] contiguous NCHW
    N, H_OUT, W_OUT, C_OUT,
    H_IN: tl.constexpr, W_IN: tl.constexpr, C_IN: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_NHW: tl.constexpr,
):
    pid_nhw = tl.program_id(0)
    pid_c   = tl.program_id(1)

    nhw_start = pid_nhw * BLOCK_NHW
    nhw_offs  = nhw_start + tl.arange(0, BLOCK_NHW)
    nhw_mask  = nhw_offs < N * H_OUT * W_OUT

    n_offs  = nhw_offs // (H_OUT * W_OUT)
    hw_offs = nhw_offs % (H_OUT * W_OUT)
    oh_offs = hw_offs // W_OUT
    ow_offs = hw_offs % W_OUT

    c_out_start = pid_c * BLOCK_C
    c_out_offs  = c_out_start + tl.arange(0, BLOCK_C)
    c_out_mask  = c_out_offs < C_OUT

    h0_offs = 2 * oh_offs - 1
    h1_offs = 2 * oh_offs
    w0_offs = 2 * ow_offs - 1
    w1_offs = 2 * ow_offs

    # Load weight tile [BLOCK_C, C_IN]
    wt_ptrs = c_out_offs[:, None] * C_IN + tl.arange(0, C_IN)[None, :]
    w_tile  = tl.load(weight_ptr + wt_ptrs, mask=c_out_mask[:, None], other=0.0)

    # Base NCHW offset per nhw (batch + channel-major part)
    inp_base = n_offs * C_IN * H_IN * W_IN   # [BLOCK_NHW]
    # Channel stride only (no spatial factor; spatial added in hw00)
    c_stride = tl.arange(0, C_IN)[None, :]  # [1, C_IN]

    hw00 = h0_offs[:, None] * W_IN + w0_offs[:, None]
    p00  = inp_base[:, None] + c_stride * (H_IN * W_IN) + hw00
    it00 = tl.load(input_ptr + p00, mask=nhw_mask[:, None], other=0.0)
    acc00 = tl.dot(w_tile, tl.trans(it00), allow_tf32=False)

    hw01 = h0_offs[:, None] * W_IN + w1_offs[:, None]
    p01  = inp_base[:, None] + c_stride * (H_IN * W_IN) + hw01
    it01 = tl.load(input_ptr + p01, mask=nhw_mask[:, None], other=0.0)
    acc01 = tl.dot(w_tile, tl.trans(it01), allow_tf32=False)

    hw10 = h1_offs[:, None] * W_IN + w0_offs[:, None]
    p10  = inp_base[:, None] + c_stride * (H_IN * W_IN) + hw10
    it10 = tl.load(input_ptr + p10, mask=nhw_mask[:, None], other=0.0)
    acc10 = tl.dot(w_tile, tl.trans(it10), allow_tf32=False)

    hw11 = h1_offs[:, None] * W_IN + w1_offs[:, None]
    p11  = inp_base[:, None] + c_stride * (H_IN * W_IN) + hw11
    it11 = tl.load(input_ptr + p11, mask=nhw_mask[:, None], other=0.0)
    acc11 = tl.dot(w_tile, tl.trans(it11), allow_tf32=False)

    result = (acc00 + acc01 + acc10 + acc11) * 0.25

    out_ptrs = c_out_offs[:, None] * H_OUT * W_OUT + oh_offs[None, :] * W_OUT + ow_offs[None, :]
    out_mask = c_out_mask[:, None] & nhw_mask[None, :]
    tl.store(output_ptr + out_ptrs, result, mask=out_mask)


@torch.fx.wrap
def fused_conv1x1_avgpool2x2(in_0, in_1):
    """
    Fused 1x1 conv + avg_pool2d(kernel=2, stride=2).
    in_0: weight [C_out, C_in, 1, 1]
    in_1: input  [N, C_in, H_in, W_in]
    Returns output [N, C_out, H_out, W_out]
    """
    N, C_IN, H_IN, W_IN = in_1.shape
    C_OUT = in_0.shape[0]
    H_OUT = H_IN // 2
    W_OUT = W_IN // 2

    output = torch.empty((N, C_OUT, H_OUT, W_OUT), dtype=in_1.dtype, device=in_1.device)

    # New kernel: grid[0] over N*H_OUT*W_OUT tiles of BLOCK_NHW positions
    grid = lambda meta: (
        triton.cdiv(N * H_OUT * W_OUT, meta['BLOCK_NHW']),
        triton.cdiv(C_OUT, meta['BLOCK_C']),
    )

    _fused_conv1x1_avgpool_kernel[grid](
        in_1, in_0, output,
        N, H_OUT, W_OUT, C_OUT,
        H_IN, W_IN, C_IN,   # constexprs
    )

    return output


def replacement_func():
    return fused_conv1x1_avgpool2x2