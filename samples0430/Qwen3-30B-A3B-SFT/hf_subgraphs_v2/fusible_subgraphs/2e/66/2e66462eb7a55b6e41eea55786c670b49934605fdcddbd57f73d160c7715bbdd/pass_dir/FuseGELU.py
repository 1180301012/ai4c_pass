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


@triton.jit
def gelu_fused_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 6291456 = 6144*1024 = 3072*2048 = 1536*4096 = 768*8192, so mask is always True
    mask = offsets < n_elements
    # Load and upcast to float32 for numerical accuracy
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    # Exact GELU: 0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x^3)))
    x3 = x_f32 * x_f32 * x_f32
    inner = 0.7978845608028654 * (x_f32 + 0.044715 * x3)
    tanh_val = tl.extra.cuda.libdevice.tanh(inner)
    out = 0.5 * x_f32 * (1.0 + tanh_val)
    # Store back in original dtype
    tl.store(out_ptr + offsets, out.to(x.dtype), mask=mask)


@torch.fx.wrap
def gelu_fused(x):
    n_elements = x.numel()
    out = torch.empty_like(x)
    # Final config: 1024 elements per block, 2 warps, 2-stage software pipeline
    # num_warps=2 → 64 threads → 16 fp16 per thread → 256-bit loads (8 x 128-bit)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    gelu_fused_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=2, num_stages=2)
    return out


def replacement_func():
    return gelu_fused