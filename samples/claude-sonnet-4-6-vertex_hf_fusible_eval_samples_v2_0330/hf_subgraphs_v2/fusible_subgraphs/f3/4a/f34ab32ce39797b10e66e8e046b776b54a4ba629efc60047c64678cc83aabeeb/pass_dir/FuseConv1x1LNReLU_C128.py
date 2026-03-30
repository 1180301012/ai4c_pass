"""
Fused pass: Conv2d(1×1) + LayerNorm(128,1,1) + ReLU
Applies when out_channels=128 (GCNet R101, normalized_shape=(128,1,1)).
BLOCK_COUT=128 (exact power of 2, no mask needed).
"""
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_CIN': 16},  num_warps=2,  num_stages=1),
        triton.Config({'BLOCK_CIN': 32},  num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_CIN': 64},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_CIN': 128}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_CIN': 256}, num_warps=8,  num_stages=2),
    ],
    key=['N', 'C_in'],
)
@triton.jit
def _fused_c128_kernel(
    x_ptr, w_ptr, b_ptr, ln_w_ptr, ln_b_ptr, out_ptr,
    N, C_in,
    BLOCK_CIN: tl.constexpr,
):
    BLOCK_COUT: tl.constexpr = 128  # C_out == 128, exact power of 2

    n = tl.program_id(0)
    cout_offs = tl.arange(0, BLOCK_COUT)

    # ---- Conv dot-product ---------------------------------------------------
    acc = tl.zeros([BLOCK_COUT], dtype=tl.float32)
    for k in range(0, tl.cdiv(C_in, BLOCK_CIN)):
        cin_offs = k * BLOCK_CIN + tl.arange(0, BLOCK_CIN)
        cin_mask = cin_offs < C_in

        x = tl.load(x_ptr + n * C_in + cin_offs,
                    mask=cin_mask, other=0.0).to(tl.float32)

        w_offs = cout_offs[:, None] * C_in + cin_offs[None, :]
        w = tl.load(w_ptr + w_offs,
                    mask=cin_mask[None, :], other=0.0).to(tl.float32)

        acc += tl.sum(w * x[None, :], axis=1)

    # conv bias
    b = tl.load(b_ptr + cout_offs).to(tl.float32)
    acc += b

    # ---- LayerNorm over 128 channels ---------------------------------------
    mean    = tl.sum(acc) / BLOCK_COUT
    diff    = acc - mean
    var     = tl.sum(diff * diff) / BLOCK_COUT
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    norm    = diff * inv_std

    ln_w = tl.load(ln_w_ptr + cout_offs).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + cout_offs).to(tl.float32)
    result = norm * ln_w + ln_b

    # ---- ReLU ---------------------------------------------------------------
    result = tl.maximum(result, 0.0)

    # ---- Store --------------------------------------------------------------
    tl.store(out_ptr + n * BLOCK_COUT + cout_offs,
             result.to(out_ptr.dtype.element_ty))


@torch.fx.wrap
def fused_conv1x1_ln_relu_c128(in_0, in_1, in_2, in_3, in_4):
    """
    in_0 : conv bias   [128]
    in_1 : conv weight [128, C_in, 1, 1]
    in_2 : ln bias     [128, 1, 1]
    in_3 : ln weight   [128, 1, 1]
    in_4 : input       [N, C_in, 1, 1]
    """
    N    = in_4.shape[0]
    C_in = in_4.shape[1]
    C_out = 128

    x    = in_4.reshape(N, C_in)
    w    = in_1.reshape(C_out, C_in)
    b    = in_0
    ln_w = in_3.reshape(C_out)
    ln_b = in_2.reshape(C_out)

    out = torch.empty(N, C_out, device=in_4.device, dtype=in_4.dtype)

    _fused_c128_kernel[(N,)](
        x, w, b, ln_w, ln_b, out,
        N, C_in,
    )

    return out.reshape(N, C_out, 1, 1)


def pattern(in_0, in_1, in_2, in_3, in_4):
    conv2d = torch.conv2d(in_4, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_5  = torch.nn.functional.layer_norm(conv2d, (128, 1, 1), in_3, in_2, 1e-05)
    tmp_6  = torch.nn.functional.relu(tmp_5, inplace=True)
    return (tmp_6,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return fused_conv1x1_ln_relu_c128