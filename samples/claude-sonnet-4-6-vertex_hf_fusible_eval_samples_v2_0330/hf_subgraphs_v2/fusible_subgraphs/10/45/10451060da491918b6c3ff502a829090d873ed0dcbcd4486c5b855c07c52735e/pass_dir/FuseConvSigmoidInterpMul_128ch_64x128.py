import torch
import triton
import triton.language as tl


# Match the final element-wise multiply (the only reliably matchable node).
def pattern(in_2, tmp_3):
    tmp_4 = in_2 * tmp_3
    return tmp_4


def replacement_args(in_2, tmp_3):
    return (in_2, tmp_3)


# Fixed-size elementwise-mul kernel for ~1M float16/bf16 elements.
# BLOCK=1024 → 1024 SMs × 1024 elements each; avoids autotune JIT overhead
# during the 25 warmup iterations.
@triton.jit
def _mul_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a * b, mask=mask)


@torch.fx.wrap
def triton_interp_mul(in_2, tmp_3):
    N    = in_2.numel()          # 1,048,576 for [1,128,64,128]
    out  = torch.empty_like(in_2)
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    _mul_kernel[grid](in_2, tmp_3, out, N, BLOCK=BLOCK)
    return out


def replacement_func():
    return triton_interp_mul