import torch
import triton
import triton.language as tl


def pattern(conv2d_out, in_2):
    tmp_3 = conv2d_out.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5


def replacement_args(conv2d_out, in_2):
    return (conv2d_out, in_2)


# ---------------------------------------------------------------------------
# Flat 1-D Triton kernel.
#
# HW is tl.constexpr so LLVM replaces  offs // HW  with a fast Barrett-
# reduction multiply-shift instead of a hardware integer-division instruction.
# BLOCK_SIZE is also constexpr so Triton generates one specialised binary per
# (HW, BLOCK_SIZE) pair — all of which are JIT-compiled and cached.
# ---------------------------------------------------------------------------
@triton.jit
def sigmoid_mul_hardtanh_flat_kernel(
    x_ptr,              # [N, C, H, W]  contiguous
    gate_ptr,           # [N, C, 1, 1]  contiguous  (NC scalars)
    out_ptr,            # [N, C, H, W]  contiguous
    total,              # N * C * H * W  (runtime int)
    HW:         tl.constexpr,  # H * W
    BLOCK_SIZE: tl.constexpr,  # elements per programme
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    # nc_idx = which (n, c) channel owns each element.
    # Because HW is constexpr LLVM turns this into a multiply-shift.
    nc_idx = offs // HW

    gate = tl.load(gate_ptr + nc_idx, mask=mask, other=0.0)
    sig  = tl.sigmoid(gate.to(tl.float32)).to(gate.dtype)

    x   = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = x * sig
    out = tl.minimum(tl.maximum(out, 0.0), 6.0)
    tl.store(out_ptr + offs, out, mask=mask)


# ---------------------------------------------------------------------------
# Threshold below which Triton kernel overhead exceeds the fusion benefit.
# N=1 cases (total ≤ ~525 K) fall through to a lightweight PyTorch fallback.
# ---------------------------------------------------------------------------
_TRITON_ELEMENT_THRESHOLD = 1_000_000

# Threshold for choosing the larger BLOCK_SIZE.
# Above this, larger blocks amortise launch overhead better on A30 (56 SMs).
_LARGE_TOTAL_THRESHOLD = 8_000_000


@torch.fx.wrap
def sigmoid_mul_hardtanh(gate, x):
    """
    gate : [N, C, 1, 1]  – conv2d SE output (broadcast gate)
    x    : [N, C, H, W]  – main feature-map
    Returns  clamp( x * sigmoid(gate), 0, 6 )
    """
    N, C, H, W = x.shape
    HW    = H * W
    NC    = N * C
    total = NC * HW

    if total >= _TRITON_ELEMENT_THRESHOLD:
        # ----------------------------------------------------------------
        # Large-batch path: fused flat-1D Triton kernel.
        # BLOCK_SIZE is chosen in Python (no autotune) to avoid unreliable
        # config selection with only 25 warmup iterations.
        # ----------------------------------------------------------------
        gate_c = gate.contiguous()
        x_c    = x.contiguous()
        out    = torch.empty_like(x_c)

        if total >= _LARGE_TOTAL_THRESHOLD:
            # float16/5: total≈16.8M  → 1026 programmes ≈ 18 waves on A30
            BS = 16384
            NW = 16
        else:
            # bfloat16/3: total≈5.7M → 698 programmes ≈ 12 waves
            # float16/4:  total≈7.5M → 915 programmes ≈ 16 waves
            BS = 8192
            NW = 16

        sigmoid_mul_hardtanh_flat_kernel[
            (triton.cdiv(total, BS),)
        ](
            x_ptr=x_c,
            gate_ptr=gate_c,
            out_ptr=out,
            total=total,
            HW=HW,
            BLOCK_SIZE=BS,
            num_warps=NW,
        )
        return out
    else:
        # ----------------------------------------------------------------
        # Small-batch path (N=1): lightweight PyTorch eager ops.
        # Avoids the Triton kernel-launch overhead which dominates for
        # tiny tensors.
        # ----------------------------------------------------------------
        out = x * gate.sigmoid()
        out.clamp_(0.0, 6.0)
        return out


def replacement_func():
    return sigmoid_mul_hardtanh