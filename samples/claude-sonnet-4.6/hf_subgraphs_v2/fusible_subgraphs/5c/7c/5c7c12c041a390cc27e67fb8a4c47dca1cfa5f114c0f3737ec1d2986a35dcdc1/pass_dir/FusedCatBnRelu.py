import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused kernel: cat([a,b],1) + BN (inference) + ReLU
# a: [B, C1, H, W],  b: [B, C2, H, W]
# output: [B, C1+C2, H, W]
#
# Uses predicated loads so only ONE of the two source tensors is read
# per thread-block, avoiding wasted bandwidth.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_HW': 512},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_HW': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_HW': 2048}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK_HW': 4096}, num_warps=32, num_stages=1),
        triton.Config({'BLOCK_HW': 8192}, num_warps=32, num_stages=1),
        triton.Config({'BLOCK_HW': 16384},num_warps=32, num_stages=1),
        triton.Config({'BLOCK_HW': 32768},num_warps=32, num_stages=1),
    ],
    key=['B', 'C1', 'C2', 'HW'],
)
@triton.jit
def fused_cat_bn_relu_kernel(
    a_ptr, b_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    B, C1, C2, HW,
    eps,
    BLOCK_HW: tl.constexpr,
):
    bc_id  = tl.program_id(0)
    hw_pid = tl.program_id(1)

    C_total = C1 + C2
    b_id = bc_id // C_total
    c_id = bc_id %  C_total

    hw_start = hw_pid * BLOCK_HW
    hw_offs  = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask  = hw_offs < HW

    # BN parameters for this output channel
    mean_val = tl.load(mean_ptr   + c_id).to(tl.float32)
    var_val  = tl.load(var_ptr    + c_id).to(tl.float32)
    w_val    = tl.load(weight_ptr + c_id).to(tl.float32)
    b_val    = tl.load(bias_ptr   + c_id).to(tl.float32)

    # Predicated loads — hardware skips the load when mask is all-False
    # so only one of {xa, xb} actually touches memory.
    use_a   = c_id < C1
    c_for_a = tl.minimum(c_id,        C1 - 1)  # clamped → always valid address
    c_for_b = tl.maximum(c_id - C1,   0)        # clamped → always valid address

    a_offs = b_id * C1 * HW + c_for_a * HW + hw_offs
    b_offs = b_id * C2 * HW + c_for_b * HW + hw_offs

    xa = tl.load(a_ptr + a_offs, mask=(hw_mask & use_a),  other=0.0)
    xb = tl.load(b_ptr + b_offs, mask=(hw_mask & ~use_a), other=0.0)

    # Exactly one of xa / xb is non-zero (the other is `other=0.0`)
    x  = xa + xb
    xf = x.to(tl.float32)

    # BN inference: y = (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = tl.rsqrt(var_val + eps)
    y = (xf - mean_val) * inv_std * w_val + b_val

    # ReLU: applying it here is memory-free (no extra IO).
    # The model's relu node runs again afterwards, but relu is idempotent.
    y = tl.maximum(y, 0.0)

    out_offs = b_id * C_total * HW + c_id * HW + hw_offs
    tl.store(out_ptr + out_offs, y.to(x.dtype), mask=hw_mask)


@torch.fx.wrap
def fused_cat_bn_relu_wrapper(a, b, running_mean, running_var, weight, bias):
    """Fuse cat([a,b],1) + BN (inference) + ReLU in a single Triton kernel."""
    B   = a.shape[0]
    C1  = a.shape[1]
    C2  = b.shape[1]
    H   = a.shape[2]
    W   = a.shape[3]
    HW  = H * W
    out = torch.empty((B, C1 + C2, H, W), dtype=a.dtype, device=a.device)
    grid = lambda meta: (B * (C1 + C2), triton.cdiv(HW, meta['BLOCK_HW']))
    fused_cat_bn_relu_kernel[grid](
        a, b,
        running_mean, running_var, weight, bias,
        out,
        B, C1, C2, HW, 1e-3,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern: cat([a, b], 1) → batch_norm(inference)
# Anchor = batch_norm (last op). relu follows in the model but is not
# matchable; it runs on our already-relu'd output (idempotent).
# ---------------------------------------------------------------------------

def pattern(a, b, running_mean, running_var, weight, bias):
    cat = torch.cat([a, b], 1)
    bn  = torch.nn.functional.batch_norm(
              cat, running_mean, running_var, weight, bias,
              False, 0.1, 0.001)
    return bn


def replacement_args(a, b, running_mean, running_var, weight, bias):
    return (a, b, running_mean, running_var, weight, bias)


def replacement_func():
    return fused_cat_bn_relu_wrapper