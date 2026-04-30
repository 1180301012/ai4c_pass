import torch
import triton
import triton.language as tl
from pass_dir.shared_kernels import dispatch_wrapper


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _fused_bn_add_relu_mean_kernel(
    in4_ptr,       # [B, C, H, W]  input feature map
    in0_ptr,       # [C]            running_mean
    in1_ptr,       # [C]            running_var
    in3_ptr,       # [C]            weight  (gamma)
    in2_ptr,       # [C]            bias    (beta)
    in5_ptr,       # [B, C, H, W]  residual
    out_ptr,       # [B, C, H, W]  relu output  (tmp6)
    mean_ptr,      # [B, C, 1, 1]  mean output  (tmp7), float32
    C, HW,
    eps: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_b = tl.program_id(0)   # batch index
    pid_c = tl.program_id(1)   # channel index

    # ── Load per-channel BN statistics (scalar) ──────────────────────────────
    rm  = tl.load(in0_ptr + pid_c).to(tl.float32)   # running_mean[c]
    rv  = tl.load(in1_ptr + pid_c).to(tl.float32)   # running_var[c]
    rg  = tl.load(in3_ptr + pid_c).to(tl.float32)   # weight  (gamma)
    rb  = tl.load(in2_ptr + pid_c).to(tl.float32)   # bias    (beta)

    # Pre-compute fused BN scale / shift:  y = x * scale + shift
    inv_std = 1.0 / tl.sqrt(rv + eps)
    scale   = rg * inv_std
    shift   = rb - rm * scale

    # ── Base offset for this (b, c) slice in the flat [B,C,H,W] layout ──────
    base = pid_b * C * HW + pid_c * HW

    # ── Accumulate sum for the spatial mean ───────────────────────────────────
    total = 0.0

    for i in range(tl.cdiv(HW, BLOCK_HW)):
        hw_off = i * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask   = hw_off < HW

        # Load input and residual (native dtype)
        x = tl.load(in4_ptr + base + hw_off, mask=mask, other=0.0)
        r = tl.load(in5_ptr + base + hw_off, mask=mask, other=0.0)

        # Upcast to fp32 for numerically stable computation
        xf   = x.to(tl.float32)
        rf   = r.to(tl.float32)

        # BN inference + add + relu
        z = tl.maximum(scale * xf + shift + rf, 0.0)

        # Write relu output in native dtype
        tl.store(out_ptr + base + hw_off, z.to(x.dtype), mask=mask)

        # Masked accumulation for mean (masked-out lanes are already 0)
        total = total + tl.sum(z)

    # ── Store the mean for this (b, c) pair ──────────────────────────────────
    tl.store(mean_ptr + pid_b * C + pid_c, (total / HW).to(tl.float32))


@torch.fx.wrap
def fused_bn_add_relu_mean(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Replaces:
        tmp4 = batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
        tmp5 = in_5 + tmp4
        tmp6 = relu(tmp5, inplace=False)
        tmp7 = tmp6.mean((2, 3), keepdim=True)
        return tmp6, tmp7
    """
    B, C, H, W = in_4.shape
    HW  = H * W
    eps = 1e-5

    # Allocate outputs
    out6    = torch.empty_like(in_4)                                  # native dtype
    out7    = torch.empty((B, C, 1, 1), dtype=torch.float32,
                          device=in_4.device)                          # float32 for mean

    # Grid: one program per (batch, channel) pair
    _fused_bn_add_relu_mean_kernel[(B, C)](
        in_4, in_0, in_1, in_3, in_2, in_5,   # in4, running_mean, running_var, weight, bias, residual
        out6, out7,
        C, HW, eps,
    )

    return out6, out7


# ── Pass 1: match BN + add (2-op fusion, relu stays outside pattern) ───────────
# Anchor = batch_norm node.  The backward walk (BN ← add ← placeholder) never
# touches relu, so relu kwargs normalization doesn't block matching.
# Single output = add result (= BN(input) + residual).
# The relu and mean operations remain in the graph and are applied normally.

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_5 = in_5 + tmp_4
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, "bn_add")


def replacement_func():
    return dispatch_wrapper