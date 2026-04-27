import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------
# Pattern: cat([x1,x2], dim=1) + batch_norm + prelu
#
# By fusing cat into the kernel we eliminate the cat intermediate tensor:
#   - No write of [N, C, H, W] cat output to DRAM
#   - No subsequent read of that cat tensor by BN
# Instead the kernel reads x1/x2 directly, saving 2 full tensor passes.
# For N=128, H=W=32 (67 MB cat tensor): saves ~144 µs of DRAM traffic.
# -----------------------------------------------------------------------

def pattern(x1, x2, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    cat_out = torch.cat([x1, x2], 1)
    bn      = torch.nn.functional.batch_norm(cat_out, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 0.001)
    out     = torch.prelu(bn, prelu_weight)
    return out


def replacement_args(x1, x2, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    return (x1, x2, running_mean, running_var, bn_weight, bn_bias, prelu_weight)


@triton.jit
def fused_cat_bn_prelu_kernel(
    x1_ptr,
    x2_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    prelu_ptr,
    out_ptr,
    C_half,          # x1.shape[1] == x2.shape[1]
    C,               # = 2 * C_half  (full output channels)
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    """
    program_id(0): output (n,c) pair  [0 .. N*C - 1]
    program_id(1): spatial tile       [0 .. ceil(HW/BLOCK_SIZE) - 1]

    Channels 0..C_half-1 come from x1; channels C_half..C-1 from x2.
    The branch on `c < C_half` is UNIFORM (same for all threads in this
    block) so no warp divergence occurs.
    """
    nc_idx    = tl.program_id(0)
    block_idx = tl.program_id(1)

    c = nc_idx % C     # full output channel index
    n = nc_idx // C    # batch index

    # Per-channel BN / PReLU parameters (fp32 for accuracy)
    mean  = tl.load(mean_ptr   + c).to(tl.float32)
    var   = tl.load(var_ptr    + c).to(tl.float32)
    gamma = tl.load(weight_ptr + c).to(tl.float32)
    beta  = tl.load(bias_ptr   + c).to(tl.float32)
    pw    = tl.load(prelu_ptr  + c).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + 1e-3)
    scale   = gamma * inv_std
    shift   = beta - mean * scale

    # Local channel index within x1 or x2 (same formula for both halves)
    c_local  = c % C_half
    src_nc   = n * C_half + c_local
    src_base = src_nc * HW

    hw_start = block_idx * BLOCK_SIZE
    offsets  = hw_start + tl.arange(0, BLOCK_SIZE)
    mask     = offsets < HW

    # Uniform branch: load from x1 (first half) or x2 (second half)
    if c < C_half:
        x = tl.load(x1_ptr + src_base + offsets, mask=mask, other=0.0).to(tl.float32)
    else:
        x = tl.load(x2_ptr + src_base + offsets, mask=mask, other=0.0).to(tl.float32)

    y   = x * scale + shift
    out = tl.where(y >= 0.0, y, pw * y)

    # Write to output [N, C, H, W] — nc_idx already encodes (n,c) for full C
    out_base = nc_idx * HW
    tl.store(out_ptr + out_base + offsets, out, mask=mask)


@torch.fx.wrap
def fused_cat_bn_prelu(x1, x2, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    N, C_half, H, W = x1.shape
    C      = C_half * 2
    HW     = H * W
    num_nc = N * C
    out    = torch.empty(N, C, H, W, dtype=x1.dtype, device=x1.device)

    num_blocks = (HW + 511) // 512

    fused_cat_bn_prelu_kernel[(num_nc, num_blocks)](
        x1, x2,
        running_mean, running_var, bn_weight, bn_bias, prelu_weight,
        out,
        C_half, C, HW,
        BLOCK_SIZE=512,
        num_warps=2,
    )
    return out


def replacement_func():
    return fused_cat_bn_prelu