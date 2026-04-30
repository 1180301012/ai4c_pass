import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 128}, num_warps=8),
        triton.Config({'BLOCK_C': 256}, num_warps=4),
        triton.Config({'BLOCK_C': 256}, num_warps=8),
        triton.Config({'BLOCK_C': 512}, num_warps=4),
        triton.Config({'BLOCK_C': 512}, num_warps=8),
    ],
    key=['C'],
)
@triton.jit
def triton_ln_kernel(
    in_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C, eps,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    c_off = tl.arange(0, BLOCK_C)
    mask = c_off < C

    x = tl.load(in_ptr + row * C + c_off, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / C
    diff = x - mean
    var  = tl.sum(diff * diff, axis=0) / C
    rstd = tl.rsqrt(var + eps)
    x_n  = diff * rstd

    w = tl.load(weight_ptr + c_off, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr   + c_off, mask=mask, other=0.0).to(tl.float32)
    out = x_n * w + b

    if IS_FP16:
        tl.store(out_ptr + row * C + c_off, out.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(out_ptr + row * C + c_off, out.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + row * C + c_off, out, mask=mask)


@torch.fx.wrap
def ln_dispatch(in_2, in_1, in_0, config):
    """
    Shared layer-norm dispatch (routing technique).
    config: "c384" | "c192" | "c96"  — ignored at runtime, only used to route in replacement_args.
    """
    C = in_2.shape[-1]
    N = in_2.numel() // C
    out = torch.empty_like(in_2)
    triton_ln_kernel[(N,)](
        in_2, in_1, in_0, out,
        N, C, 1e-5,
        IS_FP16=(in_2.dtype == torch.float16),
        IS_BF16=(in_2.dtype == torch.bfloat16),
    )
    return out