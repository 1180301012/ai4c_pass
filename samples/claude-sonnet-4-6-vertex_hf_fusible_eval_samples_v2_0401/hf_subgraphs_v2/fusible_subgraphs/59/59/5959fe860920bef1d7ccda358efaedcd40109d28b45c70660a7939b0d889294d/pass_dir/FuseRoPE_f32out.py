"""
Fused RoPE (cos/sin) kernel for TinyLlama variant:
  - input in_1: bfloat16
  - output cos and sin: float32

Fuses: cat(x,x,dim=-1) -> cos -> *1.0 -> to(float32)
       cat(x,x,dim=-1) -> sin -> *1.0 -> to(float32)

Exploits symmetry: cos(cat(x,x)) = cat(cos(x), cos(x))
                   sin(cat(x,x)) = cat(sin(x), sin(x))
"""

import torch
import triton
import triton.language as tl


def pattern(in_1):
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.float32)
    tmp_7 = tmp_5.to(dtype=torch.float32)
    return tmp_6, tmp_7


def replacement_args(in_1):
    return (in_1,)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 32},  num_warps=2),
        triton.Config({"BLOCK_K": 64},  num_warps=2),
        triton.Config({"BLOCK_K": 64},  num_warps=4),
        triton.Config({"BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_K": 256}, num_warps=4),
    ],
    key=["K"],
)
@triton.jit
def _rope_cos_sin_f32_kernel(
    X_ptr,    # [N, K] input (any dtype)
    COS_ptr,  # [N, 2K] float32 output
    SIN_ptr,  # [N, 2K] float32 output
    N, K,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K

    # Load K elements from input, convert to float32
    x = tl.load(X_ptr + row * K + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute cos and sin in float32
    c = tl.cos(x)
    s = tl.sin(x)

    # Write to first half of output [0 .. K-1]
    tl.store(COS_ptr + row * 2 * K + offsets,     c, mask=mask)
    tl.store(SIN_ptr + row * 2 * K + offsets,     s, mask=mask)
    # Write to second half of output [K .. 2K-1]
    tl.store(COS_ptr + row * 2 * K + K + offsets, c, mask=mask)
    tl.store(SIN_ptr + row * 2 * K + K + offsets, s, mask=mask)


@torch.fx.wrap
def rope_cos_sin_f32_wrapper(in_1):
    """
    in_1: [*batch, K] – frequency tensor (any dtype)
    returns: (cos_out, sin_out) each [*batch, 2K] float32
    """
    x = in_1.contiguous()
    orig_shape = x.shape          # e.g. [B, S, K]
    K = orig_shape[-1]
    N = x.numel() // K

    out_shape = list(orig_shape)
    out_shape[-1] = 2 * K         # [B, S, 2K]

    cos_out = torch.empty(out_shape, dtype=torch.float32, device=x.device)
    sin_out = torch.empty(out_shape, dtype=torch.float32, device=x.device)

    x_flat   = x.view(N, K)
    cos_flat = cos_out.view(N, 2 * K)
    sin_flat = sin_out.view(N, 2 * K)

    _rope_cos_sin_f32_kernel[(N,)](
        x_flat, cos_flat, sin_flat,
        N, K,
    )

    return cos_out, sin_out


def replacement_func():
    return rope_cos_sin_f32_wrapper