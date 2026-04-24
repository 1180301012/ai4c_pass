import torch
import triton
import triton.language as tl


@triton.jit
def bn_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C: tl.constexpr, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < C
    base = row * C
    x   = tl.load(x_ptr + base + cols, mask=mask, other=0.0)
    mean    = tl.load(mean_ptr   + cols, mask=mask, other=0.0).to(tl.float32)
    var     = tl.load(var_ptr    + cols, mask=mask, other=1.0).to(tl.float32)
    w       = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b       = tl.load(bias_ptr   + cols, mask=mask, other=0.0).to(tl.float32)
    x_f32   = x.to(tl.float32)
    inv_std = 1.0 / tl.sqrt(var + eps)
    out_f32 = (x_f32 - mean) * inv_std * w + b
    tl.store(out_ptr + base + cols, out_f32.to(x.dtype), mask=mask)


@torch.fx.wrap
def triton_batch_norm(in_0, in_1, in_2, in_3, in_4):
    # in_0: running_mean, in_1: running_var, in_2: bias, in_3: weight, in_4: input
    device = in_4.device
    dtype  = in_4.dtype
    N = in_4.shape[0]
    C = in_4.shape[1]
    mean   = in_0.to(device=device, dtype=dtype)
    var    = in_1.to(device=device, dtype=dtype)
    bias   = in_2.to(device=device, dtype=dtype)
    weight = in_3.to(device=device, dtype=dtype)
    out = torch.empty_like(in_4)
    bn_kernel[(N,)](in_4, mean, var, weight, bias, out, N, C, 1e-5, 128)
    return out


def pattern(in_0, in_1, in_2, in_3, in_4):
    return torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return triton_batch_norm