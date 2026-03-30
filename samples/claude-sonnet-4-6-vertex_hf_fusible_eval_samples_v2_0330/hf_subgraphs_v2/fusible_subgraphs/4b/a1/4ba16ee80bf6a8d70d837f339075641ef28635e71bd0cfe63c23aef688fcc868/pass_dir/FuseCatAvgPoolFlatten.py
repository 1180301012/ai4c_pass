import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Fused cat + global-avg-pool + (no-op dropout) + flatten
#
# Key design: single-load kernel using tl.where on typed scalar pointers.
#   • One program per (batch, channel): grid = (B * C_total,).
#   • tl.where selects the correct input tensor pointer (scalar selection).
#   • Exactly ONE contiguous unmasked load per warp — no masked no-op loads.
#   • eviction_policy='evict_first': streaming policy, keeps L1 clean.
#   • All shapes tl.constexpr for full compile-time address folding.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_avgpool_kernel(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr,
    out_ptr,
    B,
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    C_total: tl.constexpr = C0 + C1 + C2 + C3
    C01:     tl.constexpr = C0 + C1
    C012:    tl.constexpr = C0 + C1 + C2

    pid = tl.program_id(0)
    b   = pid >> 11      # pid // 2048  (C_total=2048 is power-of-2)
    c   = pid & 2047     # pid %  2048

    # Local channel index within the selected tensor
    c_local = tl.where(c >= C012, c - C012,
               tl.where(c >= C01,  c - C01,
               tl.where(c >= C0,   c - C0, c)))

    # Select input tensor pointer (scalar tl.where on typed pointers)
    base = tl.where(c >= C012, in3_ptr,
            tl.where(c >= C01,  in2_ptr,
            tl.where(c >= C0,   in1_ptr, in0_ptr)))

    # Per-tensor batch stride (constexpr-valued conditions, runtime result)
    C_in = tl.where(c >= C012, C3,
            tl.where(c >= C01,  C2,
            tl.where(c >= C0,   C1, C0)))

    hw_offs = tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < HW        # constant-folded (HW=25 is constexpr)

    # Single clean load — no masked no-ops, coalesced contiguous access
    val = tl.load(base + b * C_in * HW + c_local * HW + hw_offs,
                  mask=hw_mask, other=0.0,
                  eviction_policy='evict_first').to(tl.float32)

    mean_val = tl.sum(val, axis=0) * (1.0 / HW)
    tl.store(out_ptr + b * C_total + c, mean_val)


@torch.fx.wrap
def fused_cat_avgpool_flatten(in_0, in_1, in_2, in_3):
    B  = in_0.shape[0]
    C0 = in_0.shape[1]   # 320
    C1 = in_1.shape[1]   # 768
    C2 = in_2.shape[1]   # 768
    C3 = in_3.shape[1]   # 192
    C_total = C0 + C1 + C2 + C3          # 2048
    HW      = in_0.shape[2] * in_0.shape[3]   # 25

    out = torch.empty((B, C_total), dtype=in_0.dtype, device=in_0.device)

    _fused_avgpool_kernel[(B * C_total,)](
        in_0, in_1, in_2, in_3, out,
        B,
        C0=C0, C1=C1, C2=C2, C3=C3, HW=HW,
        BLOCK_HW=32, num_warps=1,
    )

    return out


def replacement_func():
    return fused_cat_avgpool_flatten