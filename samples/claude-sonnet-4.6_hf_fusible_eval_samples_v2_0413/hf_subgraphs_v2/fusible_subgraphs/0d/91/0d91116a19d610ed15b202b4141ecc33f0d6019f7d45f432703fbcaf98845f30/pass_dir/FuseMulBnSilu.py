import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Matches:
        tmp_4 = in_5 * in_4
        tmp_5 = F.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
        tmp_6 = F.silu(tmp_5, inplace=True)
        return tmp_6
    in_0 = running_mean [C]
    in_1 = running_var  [C]
    in_2 = bias         [C]
    in_3 = weight       [C]
    in_4 = sigmoid gate [B, C, 1, 1]
    in_5 = feature map  [B, C, H, W]

    Note: F.batch_norm traces through to torch.batch_norm with reordered args:
      torch.batch_norm(input, weight, bias, running_mean, running_var,
                       training, momentum, eps, cudnn_enabled)
    """
    tmp_4 = in_5 * in_4
    # torch.batch_norm arg order: input, weight, bias, running_mean, running_var, ...
    tmp_5 = torch.batch_norm(tmp_4, in_3, in_2, in_0, in_1, False, 0.1, 1e-05, True)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6


# ---------------------------------------------------------------------------
# Triton kernel  (1-D grid → contiguous x accesses)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['B', 'C', 'HW'],
)
@triton.jit
def _fused_mul_bn_silu_kernel(
    x_ptr,      # in_5: [B, C, H, W] contiguous GPU
    sc_ptr,     # in_4: [B, C, 1, 1] contiguous GPU
    mn_ptr,     # running_mean [C] GPU
    vr_ptr,     # running_var  [C] GPU
    wt_ptr,     # weight       [C] GPU
    bs_ptr,     # bias         [C] GPU
    out_ptr,    # output       [B, C, H, W]
    B, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid      = tl.program_id(0)
    offsets  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total    = B * C * HW
    mask     = offsets < total

    # Derive (b, c) from linear index over [B, C, HW]
    c = (offsets // HW) % C
    b =  offsets // (C * HW)

    # ── Load x (in_5) — contiguous sequential access ──
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_dtype = x_vals.dtype

    # ── Load scale (in_4): in_4[b, c, 0, 0] → linear idx = b*C + c ──
    sc_vals = tl.load(sc_ptr + b * C + c, mask=mask, other=0.0)

    # ── Load per-channel BN parameters (gather) ──
    mn_val = tl.load(mn_ptr + c, mask=mask, other=0.0)
    vr_val = tl.load(vr_ptr + c, mask=mask, other=0.0)
    wt_val = tl.load(wt_ptr + c, mask=mask, other=0.0)
    bs_val = tl.load(bs_ptr + c, mask=mask, other=0.0)

    # ── All arithmetic in float32 for correctness ──
    x_f32  = x_vals.to(tl.float32) * sc_vals.to(tl.float32)
    mn_f32 = mn_val.to(tl.float32)
    vr_f32 = vr_val.to(tl.float32)
    wt_f32 = wt_val.to(tl.float32)
    bs_f32 = bs_val.to(tl.float32)

    # BatchNorm (inference)
    inv_std = 1.0 / tl.sqrt(vr_f32 + eps)
    bn_out  = (x_f32 - mn_f32) * inv_std * wt_f32 + bs_f32

    # SiLU: x * σ(x)
    sig     = 1.0 / (1.0 + tl.exp(-bn_out))
    silu    = bn_out * sig

    # Cast back to original dtype and store
    tl.store(out_ptr + offsets, silu.to(x_dtype), mask=mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_mul_bn_silu(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    in_0 : running_mean [C]        (CPU tensor)
    in_1 : running_var  [C]        (CPU tensor)
    in_2 : bias         [C]        (CPU tensor)
    in_3 : weight       [C]        (CPU tensor)
    in_4 : gate         [B,C,1,1]  (CUDA tensor)
    in_5 : feature      [B,C,H,W]  (CUDA tensor)
    """
    B, C, H, W = in_5.shape
    HW     = H * W
    device = in_5.device

    # Move BN params to the same device as the activations
    mean   = in_0.to(device)
    var    = in_1.to(device)
    weight = in_3.to(device)
    bias   = in_2.to(device)

    out   = torch.empty_like(in_5)
    total = B * C * HW

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)

    _fused_mul_bn_silu_kernel[grid](
        in_5, in_4, mean, var, weight, bias, out,
        B, C, HW, 1e-5,
    )

    return out


# ---------------------------------------------------------------------------
# replacement_args / replacement_func
# ---------------------------------------------------------------------------
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return fused_mul_bn_silu