import torch
import triton
import triton.language as tl


def pattern(x, running_mean, running_var, weight, bias):
    return torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)


# No autotuning: fixed BLOCK_SIZE eliminates dict-lookup and config-selection
# overhead on every call.  For our shapes (C=384, M∈{1,32,128}):
#   M=1   : N=384  →  ceil(384 / 512) = 1 program
#   M=128 : N=49K  →  ceil(49K / 512) = 96 programs  (≈1.7 waves on A30)
@triton.jit
def _bn_inf_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    C, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N

    c   = offsets // C
    sc  = tl.load(mean_ptr  + c, mask=mask, other=0.0).to(tl.float32)
    sv  = tl.load(var_ptr   + c, mask=mask, other=1.0).to(tl.float32)
    wv  = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    bv  = tl.load(bias_ptr  + c, mask=mask, other=0.0).to(tl.float32)

    EPS    = 1e-05
    scale  = wv / tl.sqrt(sv + EPS)
    offset = bv - sc * scale

    xv   = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, (xv * scale + offset).to(xv.dtype), mask=mask)


@torch.fx.wrap
def batch_norm_triton(x, running_mean, running_var, weight, bias):
    C  = x.shape[1] if x.dim() == 2 else x.shape[-1]
    N  = x.numel()
    out = torch.empty_like(x)
    # Fixed BLOCK_SIZE=512: 2 elements/program/chan for C=384;
    # pure Python arithmetic for the grid — no lambda, no autotune overhead.
    _bn_inf_kernel[(N + 511) // 512,](
        x, running_mean, running_var, weight, bias, out,
        C, N,
        BLOCK_SIZE=512,
        num_warps=4,
    )
    return out


def replacement_func():
    return batch_norm_triton