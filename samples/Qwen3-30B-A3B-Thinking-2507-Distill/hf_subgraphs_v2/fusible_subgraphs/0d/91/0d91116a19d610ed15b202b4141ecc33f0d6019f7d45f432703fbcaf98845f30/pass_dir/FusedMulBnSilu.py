import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match: element-wise multiply -> batch_norm (inference).
    2-op pattern confirmed to match. Silu remains in the graph (applied after replacement).
    """
    tmp_4 = in_5 * in_4
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


# ---------------------------------------------------------------------------
# Triton kernel: fuse multiply + BN-inference  (no SiLU; silu stays in graph)
# Fixed BLOCK_SIZE=4096 for HW=3136, num_warps=8 for good occupancy.
# ---------------------------------------------------------------------------
@triton.jit
def fused_mul_bn_kernel(
    x_ptr,        # [B, C, H, W]  feature map
    sig_ptr,      # [B, C, 1, 1]  sigmoid input (flat bc index = b*C + c)
    mean_ptr,     # [C]
    var_ptr,      # [C]
    weight_ptr,   # [C]  gamma
    bias_ptr,     # [C]  beta
    out_ptr,      # [B, C, H, W]
    HW,
    C,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    c_idx = pid_bc % C

    # Load BN params + sigmoid as fp32 for precision
    mean_f32 = tl.load(mean_ptr   + c_idx).to(tl.float32)
    var_f32  = tl.load(var_ptr    + c_idx).to(tl.float32)
    w_f32    = tl.load(weight_ptr + c_idx).to(tl.float32)
    b_f32    = tl.load(bias_ptr   + c_idx).to(tl.float32)
    sig_f32  = tl.load(sig_ptr    + pid_bc).to(tl.float32)

    # BN affine coefficients
    inv_std = 1.0 / tl.sqrt(var_f32 + 1e-5)
    scale   = w_f32 * inv_std
    offset  = b_f32 - mean_f32 * scale

    # Sigmoid of per-(b,c) scalar in fp32
    sig_val = 1.0 / (1.0 + tl.exp(-sig_f32))

    # Spatial tile
    hw_off  = pid_hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = hw_off < HW
    base    = pid_bc * HW
    idx     = base + hw_off

    x     = tl.load(x_ptr + idx, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Fused: (x * sigmoid(sigmoid_input)) * scale + offset
    out = ((x_f32 * sig_val) * scale + offset).to(x.dtype)

    tl.store(out_ptr + idx, out, mask=mask)


@torch.fx.wrap
def fused_mul_bn(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Fused multiply + BN inference. Returns BN output (single tensor).
    in_0: running_mean, in_1: running_var, in_2: bias, in_3: weight
    in_4: sigmoid [B,C,1,1], in_5: feature map [B,C,H,W]
    """
    device = in_5.device
    B, C, H, W = in_5.shape
    HW = H * W
    BC = B * C

    mean  = in_0.to(device)
    var   = in_1.to(device)
    bias  = in_2.to(device)
    weight = in_3.to(device)

    out = torch.empty_like(in_5)

    # Use 4 warps for small BC to fit more blocks in L1; 8 warps for large BC
    n_warps = 4 if BC < 4096 else 8
    BLOCK_SIZE = 4096
    grid = (BC, triton.cdiv(HW, BLOCK_SIZE))

    fused_mul_bn_kernel[grid](
        in_5, in_4,
        mean, var, weight, bias,
        out,
        HW, C,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=n_warps,
    )

    return out


def replacement_func():
    return fused_mul_bn