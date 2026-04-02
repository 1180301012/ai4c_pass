import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


def pattern(in_0):
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return tmp_7


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# GELU kernel with libdevice.tanh — fixed optimal tile size for A30 (56 SMs)
# BLOCK_SIZE=8192: N/8192 = 768 blocks → 13.7 waves on 56 SMs
# ---------------------------------------------------------------------------
@triton.jit
def _gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load in original dtype (float16 or bfloat16)
    x = tl.load(x_ptr + offsets, mask=mask)

    # Upcast to float32 for accurate math
    x_f32 = x.to(tl.float32)

    # GELU tanh approximation using hardware libdevice tanh (SFU path):
    # GELU(x) = 0.5 * x * (1 + tanh(k*(x + 0.044715*x^3)))
    # Factored: inner = x * (k + k*0.044715 * x^2) — saves one multiplication
    # where k = 0.7978845608028654, k*0.044715 = 0.035677408136300125
    x2      = x_f32 * x_f32
    inner   = x_f32 * (0.7978845608028654 + 0.035677408136300125 * x2)
    tanh_v  = libdevice.tanh(inner)               # hardware float32 tanh
    result  = 0.5 * x_f32 * (1.0 + tanh_v)

    # Cast back to original dtype before storing
    result = result.to(x.dtype)
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def gelu_triton(in_0):
    N = in_0.numel()
    out = torch.empty_like(in_0)

    BLOCK_SIZE = 4096
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _gelu_kernel[grid](
        x_ptr=in_0,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    return out


def replacement_func():
    return gelu_triton