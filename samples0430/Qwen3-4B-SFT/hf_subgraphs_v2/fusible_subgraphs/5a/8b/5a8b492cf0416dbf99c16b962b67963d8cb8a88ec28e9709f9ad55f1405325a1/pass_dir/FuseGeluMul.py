import torch
import triton
import triton.language as tl


# ── Triton kernel ──────────────────────────────────────────────────────────────
# Use libdevice.erf (~__nv_erff) for the exact error function.
# Block size num_warps autotuned per n_elements so the compiler can
# pick matching thread counts (num_warps/32 warps = threads per block).
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE':  512}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_SIZE':  512}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=16, num_stages=1),
    ],
    key=['n_elements'],
)
@triton.jit
def _gelu_mul_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load and upcast to fp32 for numerically exact GELU
    x = tl.load(in0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(in1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    # libdevice.erf implements CUDA __nv_erff (the hf math func)
    sqrt2_inv = 0.7071067811865476
    gelu_out = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.erf(x * sqrt2_inv))

    # Fused multiply; dropout(x, training=False) is the identity
    tl.store(out_ptr + offsets, (gelu_out * y).to(x.dtype), mask=mask)


# ── Wrapper ────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def gelu_mul_triton(in_0, in_1):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _gelu_mul_kernel[grid](in_0, in_1, out, n_elements)
    return out


# ── Pattern ────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return gelu_mul_triton