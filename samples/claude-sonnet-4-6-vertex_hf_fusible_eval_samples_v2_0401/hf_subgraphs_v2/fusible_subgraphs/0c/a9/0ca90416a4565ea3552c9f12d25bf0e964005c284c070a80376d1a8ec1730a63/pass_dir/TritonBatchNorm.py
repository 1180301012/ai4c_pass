import torch
import triton
import triton.language as tl


@triton.jit
def _bn_inf_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: dim0 = N*C, dim1 = ceil(HW/BLOCK_SIZE)
    nc = tl.program_id(0)
    hw_block = tl.program_id(1)
    c = nc % C

    # Load per-channel BN parameters in float32 for numerical stability
    mean = tl.load(mean_ptr + c).to(tl.float32)
    var  = tl.load(var_ptr  + c).to(tl.float32)
    w    = tl.load(weight_ptr + c).to(tl.float32)
    b    = tl.load(bias_ptr + c).to(tl.float32)

    # Precompute affine coefficients: out = x * inv_std + bias_term
    inv_std  = w * tl.rsqrt(var + eps)
    bias_val = b - mean * inv_std

    hw_start   = hw_block * BLOCK_SIZE
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE)
    mask = hw_offsets < HW

    base = nc * HW
    x   = tl.load(x_ptr + base + hw_offsets, mask=mask, other=0.0).to(tl.float32)
    out = x * inv_std + bias_val
    tl.store(out_ptr + base + hw_offsets, out, mask=mask)


@torch.fx.wrap
def _triton_bn(in_0, in_1, in_2, in_3, in_4):
    # in_0=running_mean, in_1=running_var, in_2=bias, in_3=weight, in_4=input
    device = in_4.device

    # Move BN stats to GPU if they reside on CPU (no-op if already on device)
    mean   = in_0.to(device=device)
    var    = in_1.to(device=device)
    weight = in_3.to(device=device)
    bias   = in_2.to(device=device)

    x = in_4 if in_4.is_contiguous() else in_4.contiguous()
    N, C, H, W = x.shape
    HW  = H * W
    out = torch.empty_like(x)

    # Choose BLOCK_SIZE adaptively — power-of-2 that matches HW well
    if HW <= 64:
        BS = 64
    elif HW <= 256:
        BS = 256
    elif HW <= 512:
        BS = 512
    else:
        BS = 1024

    grid = (N * C, triton.cdiv(HW, BS))
    _bn_inf_kernel[grid](x, mean, var, weight, bias, out, C, HW, 0.001,
                         BLOCK_SIZE=BS)
    return out


# ---- Pattern: matches inference batch_norm with these exact parameters ----
def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    return tmp_5


# ---- Extract the five arguments needed by the replacement ----
def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


# ---- Return the Triton-backed replacement function ----
def replacement_func():
    return _triton_bn