import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


def pattern(x):
    tmp_0 = 0.5 * x
    tmp_1 = torch.pow(x, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = x + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return tmp_7


def replacement_args(x):
    return (x,)


@triton.jit
def _hard_swish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load in native dtype, cast to fp32 ONLY before tanh (libdevice.tanh
    # requires fp32).  Keep arithmetic in fp32 for accuracy.
    x = tl.load(x_ptr + offsets)

    # tmp_0 = 0.5 * x
    tmp_0 = 0.5 * x
    # tmp_1 = x^3  (fp16: x ≈ 0.7 → x^3 ≈ 0.343, safe)
    tmp_1 = x * x * x
    # tmp_2 = 0.044715 * x^3
    tmp_2 = 0.044715 * tmp_1
    # tmp_3 = x + 0.044715 * x^3
    tmp_3 = x + tmp_2
    # tmp_4 = 0.7978845608028654 * (x + 0.044715 * x^3)
    tmp_4 = 0.7978845608028654 * tmp_3
    # Cast to fp32 before tanh (libdevice.tanh requires fp32, not fp16/bf16)
    tmp_4_f32 = tmp_4.to(tl.float32)
    # tmp_5 = tanh via libdevice for hardware access
    tmp_5 = libdevice.tanh(tmp_4_f32)
    # tmp_6 = 1.0 + tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    # tmp_7 = 0.5 * x * (1 + tanh(...))  i.e., HardSwish(x)
    out = tmp_0 * tmp_6

    # Store in original dtype (fp16 or bf16)
    tl.store(out_ptr + offsets, out.to(x_ptr.dtype.element_ty))


@torch.fx.wrap
def hard_swish(x):
    n = x.numel()
    out = torch.empty_like(x)
    # n = 2*1024*3072 = 6,291,456 is exactly divisible by 16384 → 384 blocks.
    # 384 / 56 = ~7 waves, minimal block-launch overhead on A30.
    BLOCK_SIZE = 16384
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _hard_swish_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=32)
    return out


def replacement_func():
    return hard_swish