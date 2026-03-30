import torch
import triton
import triton.language as tl


# Pattern: scale * relu_output + bias  (the mul+add after relu)
# in_1 * tmp_2 + in_0  in the model
def pattern(a, b, c):
    return a * b + c


def replacement_args(a, b, c):
    return (a, b, c)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def scale_bias_kernel(
    a_ptr,   # [1] scale scalar  (in_1)
    b_ptr,   # [B,C,H,W] input  (tmp_2 = relu output)
    c_ptr,   # [1] bias scalar   (in_0)
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N

    a_val = tl.load(a_ptr)
    c_val = tl.load(c_ptr)
    b_val = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # Native-dtype computation matches PyTorch exactly for bfloat16
    result = a_val * b_val + c_val
    tl.store(out_ptr + offsets, result, mask=mask)


_SMALL_N_THRESHOLD = 4_000_000  # Fall back to PyTorch for small tensors


@torch.fx.wrap
def scale_bias(a, b, c):
    """Fused: a * b + c  (scale * relu_out + bias)
    Uses Triton only for bfloat16 large batches (verified correct + fast).
    Falls back to PyTorch for float16 (avoids 1-ULP FMA rounding diffs)
    and for small tensors (avoids Triton kernel-launch overhead).
    """
    N = b.numel()
    if b.dtype == torch.bfloat16 and N >= _SMALL_N_THRESHOLD:
        out = torch.empty_like(b)
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        scale_bias_kernel[grid](a, b, c, out, N)
        return out
    # PyTorch fallback: correct for all dtypes and fast for small N
    return a * b + c


def replacement_func():
    return scale_bias