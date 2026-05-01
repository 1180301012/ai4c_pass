import torch
import triton
import triton.language as tl


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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input (float16 or bfloat16)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Upcast to float32 for accurate computation
    x_f32 = x.to(tl.float32)

    # Fused GELU: 0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x^3)))
    x3 = x_f32 * x_f32 * x_f32
    inner = 0.7978845608028654 * (x_f32 + 0.044715 * x3)
    # Numerically stable tanh: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    # Clamp 2*inner to avoid float32 overflow (exp overflows at ~88.7)
    two_inner = tl.maximum(tl.minimum(2.0 * inner, 80.0), -80.0)
    e2x = tl.exp(two_inner)
    tanh_val = (e2x - 1.0) / (e2x + 1.0)
    result = 0.5 * x_f32 * (1.0 + tanh_val)

    # Cast back to original dtype and store
    tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)


@torch.fx.wrap
def gelu_wrapper(x):
    n_elements = x.numel()
    out = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    gelu_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
    )

    return out


def replacement_func():
    return gelu_wrapper