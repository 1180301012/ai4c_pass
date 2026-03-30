import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: gelu → reshape(1,124,2,768) → reshape(1,248,768)
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.reshape(1, 124, 2, 768)
    tmp_2 = tmp_1.reshape(1, 248, 768)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Triton GELU kernel.
# 190 464 = 186 × 1024  →  no partial block  →  no boundary mask needed.
# num_warps=4: 4 warps × 32 threads = 128 threads/block
#              → 1024 elements / 128 threads = 8 float16 per thread
#              → 8 × 2B = 16B = 128-bit vector load  (maximally vectorized)
# ---------------------------------------------------------------------------
@triton.jit
def _gelu_kernel(
    in_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    x     = tl.load(in_ptr + offs)          # no mask: all blocks are full
    x_f32 = x.to(tl.float32)
    cdf   = 0.5 * (1.0 + tl.math.erf(x_f32 * 0.7071067811865476))
    tl.store(out_ptr + offs, (x_f32 * cdf).to(x.dtype))   # no mask


_BLOCK = 1024
_N     = 1 * 124 * 1536   # 190 464
_GRID  = (_N // _BLOCK,)   # (186,)  — exact, no remainder


@torch.fx.wrap
def triton_gelu_reshape(in_0):
    out = torch.empty(1, 248, 768, dtype=in_0.dtype, device=in_0.device)
    _gelu_kernel[_GRID](in_0, out, BLOCK=_BLOCK, num_warps=4)
    return out


def replacement_func():
    return triton_gelu_reshape