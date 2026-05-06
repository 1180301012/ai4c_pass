import torch
import triton
import triton.language as tl


@triton.jit
def _ln_kernel(x_ptr, w_ptr, b_ptr, out_ptr, HIDDEN: tl.constexpr, eps, IS_BF16: tl.constexpr, IS_FP16: tl.constexpr):
    row  = tl.program_id(0)
    offs = tl.arange(0, HIDDEN)
    x = tl.load(x_ptr + row * HIDDEN + offs).to(tl.float32)
    mean = tl.sum(x, 0) / HIDDEN
    d    = x - mean
    var  = tl.sum(d * d, 0) / HIDDEN
    norm = d / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs).to(tl.float32)
    b = tl.load(b_ptr + offs).to(tl.float32)
    out = norm * w + b
    if IS_BF16:
        tl.store(out_ptr + row * HIDDEN + offs, out.to(tl.bfloat16))
    elif IS_FP16:
        tl.store(out_ptr + row * HIDDEN + offs, out.to(tl.float16))
    else:
        tl.store(out_ptr + row * HIDDEN + offs, out)


@torch.fx.wrap
def _diag_layer_norm(tmp_10, in_3, in_2):
    HIDDEN   = tmp_10.shape[-1]
    out      = torch.empty_like(tmp_10)
    N        = tmp_10.numel() // HIDDEN
    IS_BF16  = tmp_10.dtype == torch.bfloat16
    IS_FP16  = tmp_10.dtype == torch.float16
    _ln_kernel[(N,)](tmp_10, in_3, in_2, out, HIDDEN=HIDDEN, eps=1e-5, IS_BF16=IS_BF16, IS_FP16=IS_FP16)
    return out


# Test: does F.layer_norm alone match in the graph?
def pattern(tmp_10, in_3, in_2):
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (256,), in_3, in_2, 1e-05)
    return tmp_11


def replacement_args(tmp_10, in_3, in_2):
    return (tmp_10, in_3, in_2)


def replacement_func():
    return _diag_layer_norm