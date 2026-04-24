import torch
import triton
import triton.language as tl


def pattern(conv_out, in_2):
    tmp_3 = conv_out.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.gelu(tmp_4, approximate='none')
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
    return tmp_8


def replacement_args(conv_out, in_2):
    return (conv_out, in_2)


# Polynomial erf approximation (Abramowitz & Stegun 7.1.26), valid for all float values.
# Avoids tl.math.erf which may fault on float32 in this Triton version.
@triton.jit
def _fused_sigmoid_mul_gelu_avgpool_kernel(
    conv_out_ptr,
    in2_ptr,
    out_ptr,
    HW,
    BC,
    BLOCK_HW: tl.constexpr,
):
    pid    = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_HW)
    mask    = offsets < HW

    # Load and upcast to fp32 for accuracy (other=0.0 auto-promotes to pointer dtype)
    cv  = tl.load(conv_out_ptr + pid * HW + offsets, mask=mask, other=0.0).to(tl.float32)
    x2v = tl.load(in2_ptr      + pid * HW + offsets, mask=mask, other=0.0).to(tl.float32)

    # sigmoid
    sig  = 1.0 / (1.0 + tl.exp(-cv))
    # element-wise multiply
    prod = x2v * sig

    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    # Use polynomial erf (A&S 7.1.26): erf(x) = (1 - P(t) * exp(-x^2)) * sign(x)
    # t = 1 / (1 + 0.3275911 * |x|)
    inv_sqt2 = 0.7071067811865476          # 1 / sqrt(2)
    x2       = prod * inv_sqt2
    tx       = tl.abs(x2) * 0.3275911 + 1.0
    t        = 1.0 / tx
    poly     = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                              + t * (-1.453152027 + t * 1.061405429))))
    erf_val  = tl.where(tl.abs(x2) < 18.0,
                        (1.0 - poly * tl.exp(-x2 * x2)) * tl.where(x2 >= 0.0, 1.0, -1.0),
                        tl.where(x2 >= 0.0, 1.0, -1.0))
    gelu = prod * 0.5 * (1.0 + erf_val)

    # Masked positions (offsets >= HW) have gelu == 0 via the above formula
    avg = tl.sum(gelu) / HW

    # Cast back to input dtype using cv.dtype (known at JIT time)
    tl.store(out_ptr + pid, avg.to(cv.dtype))


@torch.fx.wrap
def fused_sigmoid_mul_gelu_avgpool(conv_out, in_2):
    B, C, H, W = in_2.shape
    HW = H * W
    BC = B * C

    # Output: [B, C]  (avgpool [B,C,1,1] + flatten(1,-1) → [B,C])
    out = torch.empty((B, C), dtype=in_2.dtype, device=in_2.device)

    _fused_sigmoid_mul_gelu_avgpool_kernel[(BC,)](
        conv_out, in_2, out,
        HW=HW,
        BC=BC,
        BLOCK_HW=256,
        num_warps=8,
    )

    return out


def replacement_func():
    return fused_sigmoid_mul_gelu_avgpool