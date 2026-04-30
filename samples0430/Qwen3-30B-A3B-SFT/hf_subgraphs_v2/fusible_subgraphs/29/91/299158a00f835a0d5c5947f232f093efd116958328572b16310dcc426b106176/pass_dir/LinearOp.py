import torch
import triton
import triton.language as tl


@triton.jit
def _linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N_out, K,
    BLOCK_K: tl.constexpr,
):
    # One program per output element (m, n)
    m = tl.program_id(0)
    n = tl.program_id(1)

    # Dummy load just to capture the input pointer's element dtype
    x_bad = tl.load(input_ptr, mask=False, other=0.0)

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs = k_start + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(input_ptr  + m * K + offs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + n * K + offs, mask=mask, other=0.0)
        acc += x.to(tl.float32) * w.to(tl.float32)

    dot  = tl.sum(acc, axis=0)
    bias = tl.load(bias_ptr + n).to(tl.float32)
    tl.store(output_ptr + m * N_out + n, (dot + bias).to(x_bad.dtype))


@torch.fx.wrap
def triton_linear(input, weight, bias):
    # Use only metadata properties — NO aten ops allowed by PoisonDispatchTensor
    ndim   = input.dim()       # 1-D [K] or 2-D [B, K] — metadata, no dispatch
    N_out  = weight.shape[0]   # output feature dim (always 2)
    if ndim == 1:
        M   = 1
        K   = input.shape[0]   # [K] → K elements
        out = torch.empty((M, N_out), dtype=input.dtype, device=input.device)
    else:
        M   = input.shape[0]   # [B, K] → batch dim
        K   = input.shape[1]   # [B, K] → feature dim
        out = torch.empty((M, N_out), dtype=input.dtype, device=input.device)
    grid = lambda meta: (M, N_out)
    _linear_kernel[grid](input, weight, bias, out, M, N_out, K, BLOCK_K=512)
    return out


def pattern(bias, weight, input):
    return torch.nn.functional.linear(input, weight, bias)


def replacement_args(bias, weight, input):
    return (bias, weight, input)


def replacement_func():
    return triton_linear