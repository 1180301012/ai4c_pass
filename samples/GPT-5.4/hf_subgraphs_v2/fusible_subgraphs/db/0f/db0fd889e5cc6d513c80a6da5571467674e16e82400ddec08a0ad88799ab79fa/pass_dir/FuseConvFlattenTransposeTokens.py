import torch
import triton
import triton.language as tl


def pattern(in_0, in_2, in_1):
    conv2d = torch.conv2d(in_0, in_2, in_1, (2, 2), (0, 0), (1, 1), 1)
    tmp_8 = conv2d.flatten(2)
    tmp_9 = tmp_8.transpose(1, 2)
    return tmp_9


def replacement_args(in_0, in_2, in_1):
    return (in_0, in_2, in_1)


@triton.jit
def _conv_tokens_kernel(inp_ptr, weight_ptr, bias_ptr, out_ptr):
    pid = tl.program_id(0)  # token index: 0..224
    c = tl.arange(0, 32)

    oh = pid // 15
    ow = pid % 15
    ih = oh * 2
    iw = ow * 2

    acc = tl.load(bias_ptr + c).to(tl.float32)

    # inp: [1, 3, 30, 30] contiguous, strides [2700, 900, 30, 1]
    # weight: [32, 3, 2, 2] contiguous, strides [12, 4, 2, 1]
    for ic in range(3):
        for kh in range(2):
            for kw in range(2):
                inp_off = ic * 900 + (ih + kh) * 30 + (iw + kw)
                inp_val = tl.load(inp_ptr + inp_off).to(tl.float32)
                w_off = c * 12 + ic * 4 + kh * 2 + kw
                w_val = tl.load(weight_ptr + w_off).to(tl.float32)
                acc += inp_val * w_val

    tl.store(out_ptr + pid * 32 + c, acc)


@torch.fx.wrap
def conv_tokens_triton(in_0, in_2, in_1):
    out = torch.empty((1, 225, 32), device=in_0.device, dtype=in_0.dtype)
    _conv_tokens_kernel[(225,)](
        inp_ptr=in_0,
        weight_ptr=in_2,
        bias_ptr=in_1,
        out_ptr=out,
        num_warps=1,
    )
    return out


def replacement_func():
    return conv_tokens_triton