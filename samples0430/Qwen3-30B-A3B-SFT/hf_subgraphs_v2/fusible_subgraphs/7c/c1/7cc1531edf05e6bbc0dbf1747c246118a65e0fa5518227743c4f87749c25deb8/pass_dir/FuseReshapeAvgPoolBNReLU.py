import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: reshape → avg_pool2d → batch_norm (inference) → silu
# Shapes (both bfloat16 and float16 graphs share the same structure):
#   in_4  : [4, 128, 256]   CUDA
#   in_0  : [512]           CPU  running_mean
#   in_1  : [512]           CPU  running_var
#   in_2  : [512]           CPU  bias
#   in_3  : [512]           CPU  weight
#   output: [1, 512, 8, 8]  CUDA
# ─────────────────────────────────────────────────────────────────────────────


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = in_4.reshape(1, 512, 16, 16)
    tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.silu(tmp_6, inplace=True)
    return (tmp_7,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused avg-pool(2×2) + BN-inference + SiLU
#
# Grid  : (C,)          – one program per channel
# Thread: BLOCK_SIZE = H_out × W_out = 8 × 8 = 64 threads per program
#
# Each thread handles one (oh, ow) spatial position for channel c.
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=8),
    ],
    key=['N_OUT'],
)
@triton.jit
def _fused_avgpool_bn_silu_kernel(
    in_ptr,      # [1, C, H_in, W_in]  – reshaped flat view of in_4
    rm_ptr,      # [C]  running_mean
    rv_ptr,      # [C]  running_var
    w_ptr,       # [C]  BN weight (gamma)
    b_ptr,       # [C]  BN bias   (beta)
    out_ptr,     # [1, C, H_out, W_out]
    eps,         # BN epsilon (float scalar)
    H_IN:   tl.constexpr,   # 16
    W_IN:   tl.constexpr,   # 16
    H_OUT:  tl.constexpr,   # 8
    W_OUT:  tl.constexpr,   # 8
    N_OUT:  tl.constexpr,   # H_OUT * W_OUT = 64
    BLOCK_SIZE: tl.constexpr,  # must equal N_OUT
):
    c = tl.program_id(0)

    # ── BN inference: precompute scale and shift ──────────────────────────────
    rm      = tl.load(rm_ptr + c).to(tl.float32)
    rv      = tl.load(rv_ptr + c).to(tl.float32)
    w       = tl.load(w_ptr  + c).to(tl.float32)
    b       = tl.load(b_ptr  + c).to(tl.float32)
    inv_std = 1.0 / tl.sqrt(rv + eps)
    scale   = w * inv_std
    shift   = b - rm * scale

    # ── Process all N_OUT output spatial positions ────────────────────────────
    idx     = tl.arange(0, BLOCK_SIZE)          # [0 .. 63]
    oh      = idx // W_OUT
    ow      = idx % W_OUT

    ih0 = oh * 2
    ih1 = oh * 2 + 1
    iw0 = ow * 2
    iw1 = ow * 2 + 1

    base_in = c * (H_IN * W_IN)                 # channel stride in in_ptr

    x00 = tl.load(in_ptr + base_in + ih0 * W_IN + iw0).to(tl.float32)
    x01 = tl.load(in_ptr + base_in + ih0 * W_IN + iw1).to(tl.float32)
    x10 = tl.load(in_ptr + base_in + ih1 * W_IN + iw0).to(tl.float32)
    x11 = tl.load(in_ptr + base_in + ih1 * W_IN + iw1).to(tl.float32)

    avg = (x00 + x01 + x10 + x11) * 0.25

    # ── Fused BN + SiLU ────────────────────────────────────────────────────────
    val   = avg * scale + shift
    out   = val * tl.sigmoid(val)               # SiLU = x * σ(x)

    base_out = c * N_OUT
    tl.store(out_ptr + base_out + idx, out.to(x00.dtype))


# ─────────────────────────────────────────────────────────────────────────────
# Python wrapper (must be @torch.fx.wrap)
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_avgpool_bn_silu(in_0, in_1, in_2, in_3, in_4):
    """
    in_0 : running_mean  [C]  (may be CPU)
    in_1 : running_var   [C]  (may be CPU)
    in_2 : bias          [C]  (may be CPU)
    in_3 : weight        [C]  (may be CPU)
    in_4 : input         [4,128,256]  CUDA
    """
    C     = 512
    H_OUT = 8
    W_OUT = 8
    N_OUT = H_OUT * W_OUT   # 64

    device = in_4.device
    # Move BN parameters to the same device as the activation tensor
    rm  = in_0.to(device)
    rv  = in_1.to(device)
    b   = in_2.to(device)
    w   = in_3.to(device)

    # Reshape in_4 to [1, C, H_in, W_in] so that channel stride is C*H_in*W_in
    x   = in_4.reshape(1, C, 16, 16)
    out = torch.empty((1, C, H_OUT, W_OUT), dtype=in_4.dtype, device=device)

    _fused_avgpool_bn_silu_kernel[(C,)](
        x, rm, rv, w, b, out,
        1e-5,          # eps
        16, 16,        # H_IN, W_IN
        H_OUT, W_OUT,  # 8, 8
        N_OUT,         # 64
        # BLOCK_SIZE provided by autotune
    )

    return (out,)


def replacement_func():
    return fused_avgpool_bn_silu