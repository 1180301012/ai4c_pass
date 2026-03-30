"""
Pass: fuse depthwise conv2d(stride=1, groups=768) + mean((2,3)) into one Triton kernel.
Fully self-contained — no external pass_dir imports.
"""
import operator
import torch
import triton
import triton.language as tl


@triton.jit
def _dw_s1_g768_kernel(
    X, W, Out, SumBuf,
    B, C, H_in, W_in, H_out, W_out,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    bc = tl.program_id(0)
    b  = bc // C
    c  = bc % C
    x_base   = b * C * H_in * W_in   + c * H_in * W_in
    out_base = b * C * H_out * W_out + c * H_out * W_out
    w_pos  = tl.arange(0, BLOCK_W)
    w_mask = w_pos < W_out
    acc_sum = tl.zeros([BLOCK_W], dtype=tl.float32)
    for h_pos in range(H_out):
        acc = tl.zeros([BLOCK_W], dtype=tl.float32)
        for kh in range(3):
            ih   = h_pos + kh - 1
            h_ok = (ih >= 0) & (ih < H_in)
            for kw in range(3):
                iw   = w_pos + kw - 1
                w_ok = (iw >= 0) & (iw < W_in)
                wgt  = tl.load(W + c * 9 + kh * 3 + kw).to(tl.float32)
                xv   = tl.load(X + x_base + ih * W_in + iw,
                               mask=w_mask & h_ok & w_ok, other=0.0).to(tl.float32)
                acc  = acc + xv * wgt
        if IS_FP16:
            tl.store(Out + out_base + h_pos * W_out + w_pos, acc.to(tl.float16), mask=w_mask)
        elif IS_BF16:
            tl.store(Out + out_base + h_pos * W_out + w_pos, acc.to(tl.bfloat16), mask=w_mask)
        else:
            tl.store(Out + out_base + h_pos * W_out + w_pos, acc, mask=w_mask)
        acc_sum = acc_sum + tl.where(w_mask, acc, tl.zeros([BLOCK_W], dtype=tl.float32))
    tl.store(SumBuf + bc, tl.sum(acc_sum))


@torch.fx.wrap
def _dw_s1_g768_call(in_0, in_1):
    x = in_1
    w = in_0.to(device=x.device, dtype=x.dtype)
    B, C, H_in, W_in = x.shape
    H_out, W_out = H_in, W_in
    out     = torch.empty((B, C, H_out, W_out), dtype=x.dtype, device=x.device)
    sum_buf = torch.empty(B * C, dtype=torch.float32, device=x.device)
    IS_FP16 = (x.dtype == torch.float16)
    IS_BF16 = (x.dtype == torch.bfloat16)
    _dw_s1_g768_kernel[(B * C,)](x, w, out, sum_buf, B, C, H_in, W_in, H_out, W_out, IS_FP16=IS_FP16, IS_BF16=IS_BF16, BLOCK_W=64)
    mean_out = (sum_buf / (H_out * W_out)).to(x.dtype).view(B, C, 1, 1)
    return out, mean_out


def kernel_wrapper_s1_g768(in_0, in_1):
    result   = _dw_s1_g768_call(in_0, in_1)
    conv_out = result[0]
    mean_out = result[1]
    return (conv_out, mean_out)


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 768)
    tmp_2 = conv2d.mean((2, 3), keepdim=True)
    return (conv2d, tmp_2)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return kernel_wrapper_s1_g768