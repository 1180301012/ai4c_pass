import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'num_warps': 4, 'num_stages': 2}),
        triton.Config({'num_warps': 8, 'num_stages': 2}),
        triton.Config({'num_warps': 16, 'num_stages': 2}),
        triton.Config({'num_warps': 4, 'num_stages': 3}),
        triton.Config({'num_warps': 8, 'num_stages': 3}),
        triton.Config({'num_warps': 16, 'num_stages': 3}),
    ],
    key=['H'],
)
@triton.jit
def _sub_pow_sum_mul_kernel(
    in1_ptr, in2_ptr, sc_ptr, out_ptr,
    H,
    K_CONST: tl.constexpr,
    D_CONST: tl.constexpr,
    IS_FP16: tl.constexpr,
):
    # Fuses: (in1 - in2).pow(2).sum(dim=3) * scale
    h = tl.program_id(0)
    k_offsets = tl.arange(0, K_CONST)
    d_offsets = tl.arange(0, D_CONST)
    in1_addr = h * K_CONST * D_CONST + k_offsets[:, None] * D_CONST + d_offsets[None, :]
    in1_vals = tl.load(in1_ptr + in1_addr).to(tl.float32)   # [K, D]
    in2_addr = k_offsets[:, None] * D_CONST + d_offsets[None, :]
    in2_vals = tl.load(in2_ptr + in2_addr).to(tl.float32)   # [K, D]
    diff = in1_vals - in2_vals                               # [K, D]
    dist = tl.sum(diff * diff, axis=1)                       # [K]
    sc_vals = tl.load(sc_ptr + k_offsets).to(tl.float32)
    scaled = sc_vals * dist
    out_addr = h * K_CONST + k_offsets
    if IS_FP16:
        tl.store(out_ptr + out_addr, scaled.to(tl.float16))
    else:
        tl.store(out_ptr + out_addr, scaled.to(tl.bfloat16))


@triton.autotune(
    configs=[
        triton.Config({'num_warps': 4, 'num_stages': 2}),
        triton.Config({'num_warps': 8, 'num_stages': 2}),
        triton.Config({'num_warps': 16, 'num_stages': 2}),
        triton.Config({'num_warps': 4, 'num_stages': 3}),
        triton.Config({'num_warps': 8, 'num_stages': 3}),
        triton.Config({'num_warps': 16, 'num_stages': 3}),
    ],
    key=['H'],
)
@triton.jit
def _pow_sum_mul_kernel(
    x_ptr, sc_ptr, out_ptr,
    H,
    K_CONST: tl.constexpr,
    D_CONST: tl.constexpr,
    IS_FP16: tl.constexpr,
):
    h = tl.program_id(0)
    k_offsets = tl.arange(0, K_CONST)
    d_offsets = tl.arange(0, D_CONST)
    x_addr = h * K_CONST * D_CONST + k_offsets[:, None] * D_CONST + d_offsets[None, :]
    x_vals = tl.load(x_ptr + x_addr).to(tl.float32)
    dist = tl.sum(x_vals * x_vals, axis=1)
    sc_vals = tl.load(sc_ptr + k_offsets).to(tl.float32)
    scaled = sc_vals * dist
    out_addr = h * K_CONST + k_offsets
    if IS_FP16:
        tl.store(out_ptr + out_addr, scaled.to(tl.float16))
    else:
        tl.store(out_ptr + out_addr, scaled.to(tl.bfloat16))


@triton.autotune(
    configs=[
        triton.Config({'num_warps': 4, 'num_stages': 2}),
        triton.Config({'num_warps': 8, 'num_stages': 2}),
        triton.Config({'num_warps': 16, 'num_stages': 2}),
        triton.Config({'num_warps': 4, 'num_stages': 3}),
        triton.Config({'num_warps': 8, 'num_stages': 3}),
        triton.Config({'num_warps': 16, 'num_stages': 3}),
    ],
    key=['H'],
)
@triton.jit
def _expand_sub_kernel(
    in0_ptr, in4_ptr, out_ptr,
    H,
    K_CONST: tl.constexpr,
    D_CONST: tl.constexpr,
):
    h = tl.program_id(0)
    k_offsets = tl.arange(0, K_CONST)
    d_offsets = tl.arange(0, D_CONST)
    in4_addr = h * D_CONST + d_offsets
    in4_vals = tl.load(in4_ptr + in4_addr)
    in0_addr = k_offsets[:, None] * D_CONST + d_offsets[None, :]
    in0_vals = tl.load(in0_ptr + in0_addr)
    result = tl.expand_dims(in4_vals, 0) - in0_vals
    out_addr = h * K_CONST * D_CONST + k_offsets[:, None] * D_CONST + d_offsets[None, :]
    tl.store(out_ptr + out_addr, result)


@torch.fx.wrap
def shared_dispatch(a, b, route, c=None):
    if route == "sub_pow_sum_mul":
        # a: in_1 [1,H,K,D], b: in_2 [1,1,K,D], c: in_3 [1,1,K]
        _, H, K, D = a.shape
        out = torch.empty((1, H, K), dtype=a.dtype, device=a.device)
        IS_FP16 = (a.dtype == torch.float16)
        _sub_pow_sum_mul_kernel[(H,)](a, b, c, out, H, K_CONST=32, D_CONST=512, IS_FP16=IS_FP16)
        return out
    elif route == "pow_sum_mul":
        # a: x [1,H,K,D], b: scale [1,1,K]
        _, H, K, D = a.shape
        out = torch.empty((1, H, K), dtype=a.dtype, device=a.device)
        IS_FP16 = (a.dtype == torch.float16)
        _pow_sum_mul_kernel[(H,)](a, b, out, H, K_CONST=32, D_CONST=512, IS_FP16=IS_FP16)
        return out
    elif route == "expand_sub":
        # a: in_0 [K,D], b: in_4 [1,H,D]
        _, H, D = b.shape
        K = 32
        out = torch.empty((1, H, K, D), dtype=b.dtype, device=b.device)
        _expand_sub_kernel[(H,)](a, b, out, H, K_CONST=32, D_CONST=512)
        return out
    else:
        return a