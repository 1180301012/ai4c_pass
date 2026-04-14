import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------ #
#  Integer dtype codes – safe as tl.constexpr and in dicts            #
# ------------------------------------------------------------------ #
_BF16 = 0
_F16  = 1
_F32  = 2

_DTYPE_CODE = {
    torch.bfloat16: _BF16,
    torch.float16:  _F16,
    torch.float32:  _F32,
}

# ================================================================== #
#  Trig + cast kernel  (cos or sin, any in-dtype, any out-dtype)       #
#  Pattern:  tensor -> .cos()/.sin() -> * 1.0 -> .to(dtype)           #
#                                                                      #
#  IN_CODE  : integer code of the INPUT tensor dtype                   #
#  OUT_CODE : integer code of the OUTPUT dtype                         #
#  IS_COS   : True → cos, False → sin                                 #
# ================================================================== #

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=32),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=32),
    ],
    key=['N'],
)
@triton.jit
def _trig_cast_kernel(
    in_ptr, out_ptr, N,
    IS_COS:   tl.constexpr,
    IN_CODE:  tl.constexpr,   # 0=bf16  1=f16  2=f32
    OUT_CODE: tl.constexpr,   # 0=bf16  2=f32
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N

    val_f32 = tl.load(in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if IS_COS:
        result_f32 = tl.cos(val_f32)
    else:
        result_f32 = tl.sin(val_f32)

    # ---- intermediate cast matching PyTorch reference behavior ----
    # PyTorch stores the trig result in the INPUT tensor's dtype before
    # the explicit .to(output_dtype) conversion; reproduce that here.
    if IN_CODE == 0:   # bfloat16
        inter = result_f32.to(tl.bfloat16)
    elif IN_CODE == 1:  # float16
        inter = result_f32.to(tl.float16)
    else:               # float32
        inter = result_f32

    # ---- final output cast ----
    if OUT_CODE == 0:  # bfloat16
        out = inter.to(tl.bfloat16)
    else:              # float32
        out = inter.to(tl.float32)

    tl.store(out_ptr + offsets, out, mask=mask)


def _trig_impl(x, is_cos, out_code):
    """Launch _trig_cast_kernel for element-wise cos/sin + cast."""
    N         = x.numel()
    in_code   = _DTYPE_CODE[x.dtype]
    out_dtype = torch.bfloat16 if out_code == _BF16 else torch.float32
    out       = torch.empty(x.shape, dtype=out_dtype, device=x.device)

    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _trig_cast_kernel[grid](
        in_ptr=x, out_ptr=out, N=N,
        IS_COS=is_cos, IN_CODE=in_code, OUT_CODE=out_code,
    )
    return out


# ================================================================== #
#  RMSNorm fused kernel                                                #
#  in_2.to(f32) -> pow(2) -> mean -> +eps -> rsqrt -> mul -> cast      #
#  -> weight * norm                                                    #
# ================================================================== #

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=32),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=32, num_stages=2),
    ],
    key=['D'],
)
@triton.jit
def _rmsnorm_kernel_bf16(
    weight_ptr, input_ptr, output_ptr,
    D, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx   = tl.program_id(0)
    row_start = row_idx * D
    offsets   = tl.arange(0, BLOCK_SIZE)
    mask      = offsets < D

    x      = tl.load(input_ptr  + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets,              mask=mask, other=1.0).to(tl.float32)

    x_sq_mean = tl.sum(x * x, axis=0) / D
    rsqrt_val = tl.rsqrt(x_sq_mean + eps)
    out       = weight * (x * rsqrt_val)
    tl.store(output_ptr + row_start + offsets, out.to(tl.bfloat16), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=32),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=32, num_stages=2),
    ],
    key=['D'],
)
@triton.jit
def _rmsnorm_kernel_f32(
    weight_ptr, input_ptr, output_ptr,
    D, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx   = tl.program_id(0)
    row_start = row_idx * D
    offsets   = tl.arange(0, BLOCK_SIZE)
    mask      = offsets < D

    x      = tl.load(input_ptr  + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets,              mask=mask, other=1.0).to(tl.float32)

    x_sq_mean = tl.sum(x * x, axis=0) / D
    rsqrt_val = tl.rsqrt(x_sq_mean + eps)
    out       = weight * (x * rsqrt_val)
    tl.store(output_ptr + row_start + offsets, out.to(tl.float32), mask=mask)


def _rmsnorm_bf16_impl(in_0, in_2):
    orig_shape = in_2.shape
    D = orig_shape[-1]
    N = in_2.numel() // D
    out = torch.empty(orig_shape, dtype=torch.bfloat16, device=in_2.device)
    _rmsnorm_kernel_bf16[(N,)](
        weight_ptr=in_0, input_ptr=in_2, output_ptr=out, D=D, eps=1e-06,
    )
    return out


def _rmsnorm_f32_impl(in_0, in_2):
    orig_shape = in_2.shape
    D = orig_shape[-1]
    N = in_2.numel() // D
    out = torch.empty(orig_shape, dtype=torch.float32, device=in_2.device)
    _rmsnorm_kernel_f32[(N,)](
        weight_ptr=in_0, input_ptr=in_2, output_ptr=out, D=D, eps=1e-05,
    )
    return out


# ================================================================== #
#  Single shared dispatch wrapper (all passes return this same object) #
# ================================================================== #

@torch.fx.wrap
def shared_dispatch(a, b_or_route, maybe_route=None):
    """
    Routing convention
    ------------------
    Trig passes:    shared_dispatch(tensor,       route_str)
    RMSNorm passes: shared_dispatch(in_0,  in_2,  route_str)

    Trig routes:  "cos_bf16", "sin_bf16", "cos_f32", "sin_f32"
    RMSNorm routes: "rmsnorm_1e6_bf16", "rmsnorm_1e5_f32"
    """
    if maybe_route is None:
        route = b_or_route
        if route == "cos_bf16":
            return _trig_impl(a, True,  _BF16)
        elif route == "sin_bf16":
            return _trig_impl(a, False, _BF16)
        elif route == "cos_f32":
            return _trig_impl(a, True,  _F32)
        else:                          # "sin_f32"
            return _trig_impl(a, False, _F32)
    else:
        route = maybe_route
        if route == "rmsnorm_1e6_bf16":
            return _rmsnorm_bf16_impl(a, b_or_route)
        else:                          # "rmsnorm_1e5_f32"
            return _rmsnorm_f32_impl(a, b_or_route)