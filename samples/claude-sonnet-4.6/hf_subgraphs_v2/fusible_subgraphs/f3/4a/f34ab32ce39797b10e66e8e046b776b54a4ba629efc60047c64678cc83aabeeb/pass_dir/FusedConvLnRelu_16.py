import torch
import triton
import triton.language as tl


@triton.jit
def _fused_conv1x1_ln_relu_kernel(
    x_ptr, w_ptr, cb_ptr, ln_w_ptr, ln_b_ptr, out_ptr,
    N, C_in, C_out,
    BLOCK_COUT: tl.constexpr,
    BLOCK_CIN: tl.constexpr,
):
    n = tl.program_id(0)
    c_idx = tl.arange(0, BLOCK_COUT)
    c_mask = c_idx < C_out
    c_clamped = tl.where(c_mask, c_idx, 0)

    acc = tl.zeros((BLOCK_COUT,), dtype=tl.float32)
    for k_start in range(0, C_in, BLOCK_CIN):
        k_idx = k_start + tl.arange(0, BLOCK_CIN)
        k_mask = k_idx < C_in
        k_clamped = tl.where(k_mask, k_idx, 0)
        x_vals = tl.load(x_ptr + n * C_in + k_clamped, mask=k_mask, other=0.0).to(tl.float32)
        w_vals = tl.load(
            w_ptr + c_clamped[:, None] * C_in + k_clamped[None, :],
            mask=c_mask[:, None] & k_mask[None, :], other=0.0
        ).to(tl.float32)
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    cb_vals = tl.load(cb_ptr + c_clamped, mask=c_mask, other=0.0).to(tl.float32)
    acc += cb_vals

    acc_valid = tl.where(c_mask, acc, 0.0)
    mean = tl.sum(acc_valid) / C_out
    centered = acc - mean
    centered_sq = tl.where(c_mask, centered * centered, 0.0)
    var = tl.sum(centered_sq) / C_out
    inv_std = 1.0 / tl.sqrt(var + 1e-5)

    ln_w = tl.load(ln_w_ptr + c_clamped, mask=c_mask, other=1.0).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + c_clamped, mask=c_mask, other=0.0).to(tl.float32)
    out_val = tl.maximum(centered * inv_std * ln_w + ln_b, 0.0)
    tl.store(out_ptr + n * C_out + c_idx, out_val, mask=c_mask)


@torch.fx.wrap
def _fused_wrapper(in_0, in_1, in_2, in_3, in_4):
    N = in_4.shape[0]
    C_in = in_1.shape[1]
    C_out = in_1.shape[0]
    BLOCK_COUT = triton.next_power_of_2(C_out)
    BLOCK_COUT = max(BLOCK_COUT, 16)
    BLOCK_CIN = triton.next_power_of_2(min(C_in, 512))
    BLOCK_CIN = max(BLOCK_CIN, 32)
    out = torch.empty((N, C_out, 1, 1), dtype=in_4.dtype, device=in_4.device)
    _fused_conv1x1_ln_relu_kernel[(N,)](
        in_4, in_1, in_0, in_3, in_2, out,
        N, C_in, C_out,
        BLOCK_COUT=BLOCK_COUT, BLOCK_CIN=BLOCK_CIN,
    )
    return out


def pattern(in_0, in_1, in_2, in_3, in_4):
    conv2d = torch.conv2d(in_4, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_5 = torch.nn.functional.layer_norm(conv2d, (16, 1, 1), in_3, in_2, 1e-05)
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=True)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return _fused_wrapper