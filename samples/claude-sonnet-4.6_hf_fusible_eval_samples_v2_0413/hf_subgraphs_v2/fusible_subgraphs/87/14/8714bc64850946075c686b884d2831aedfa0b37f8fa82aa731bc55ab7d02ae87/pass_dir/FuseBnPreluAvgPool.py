import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel: fused BN (inference) + PReLU — one program per (batch, channel)
# ---------------------------------------------------------------------------
@triton.jit
def _bn_prelu_fwd(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, prelu_ptr,
    out_ptr,
    C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    c    = pid % C
    base = pid * HW

    # Load per-channel BN/PReLU params (upcast to fp32 for precision)
    mean_v  = tl.load(mean_ptr   + c).to(tl.float32)
    var_v   = tl.load(var_ptr    + c).to(tl.float32)
    w_v     = tl.load(weight_ptr + c).to(tl.float32)
    b_v     = tl.load(bias_ptr   + c).to(tl.float32)
    prelu_v = tl.load(prelu_ptr  + c).to(tl.float32)

    # Fused BN transform: y = x * scale + shift
    inv_std = 1.0 / tl.sqrt(var_v + 0.001)
    scale   = w_v * inv_std
    shift   = b_v - mean_v * scale

    for i in range(0, HW, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < HW

        xv  = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        y   = xv * scale + shift
        out = tl.where(y >= 0.0, y, prelu_v * y)   # PReLU
        # Store (Triton auto-casts fp32 → output tensor dtype)
        tl.store(out_ptr + base + offs, out, mask=mask)


# ---------------------------------------------------------------------------
# Pattern: batch_norm (inference) → prelu — single output
# ---------------------------------------------------------------------------
def pattern(x, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    bn_out    = torch.nn.functional.batch_norm(
        x, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 0.001
    )
    prelu_out = torch.prelu(bn_out, prelu_weight)
    return prelu_out


def replacement_args(x, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    return (x, running_mean, running_var, bn_weight, bn_bias, prelu_weight)


# ---------------------------------------------------------------------------
# Replacement wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _bn_prelu_wrapper(x, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    B, C, H, W = x.shape
    HW = H * W

    # BLOCK_SIZE: next power-of-2 ≥ HW, capped at 4096
    # Default num_warps=4 (Triton default) gives best occupancy for BLOCK_SIZE=4096
    BLOCK_SIZE = min(4096, triton.next_power_of_2(HW))

    out = torch.empty_like(x)

    _bn_prelu_fwd[(B * C,)](
        x, running_mean, running_var, bn_weight, bn_bias, prelu_weight,
        out,
        C, HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return _bn_prelu_wrapper