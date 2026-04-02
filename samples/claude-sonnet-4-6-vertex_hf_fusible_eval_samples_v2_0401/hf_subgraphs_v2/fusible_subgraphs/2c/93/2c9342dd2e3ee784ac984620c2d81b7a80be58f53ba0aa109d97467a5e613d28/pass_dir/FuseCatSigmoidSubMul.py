import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: cat([a, b, c], dim=2) → sigmoid → sub 0.25 → mul pi
# a=(B,1,6400), b=(B,1,1600), c=(B,1,400) → out=(B,1,8400)
# ---------------------------------------------------------------------------

def pattern(a, b, c):
    cat = torch.cat([a, b, c], 2)
    sig = cat.sigmoid()
    sub = sig - 0.25
    mul = sub * 3.141592653589793
    return mul


def replacement_args(a, b, c):
    return (a, b, c)


# ---------------------------------------------------------------------------
# Single fused kernel with BLOCK_SIZE=8192.
# For B=24: grid=(24,2)=48 programs < 56 SMs → fits in ONE WAVE.
# For B=32: grid=(32,2)=64 programs ≈ 1 wave.
# This eliminates multi-wave scheduling overhead vs. smaller BLOCK_SIZE.
# ---------------------------------------------------------------------------

@triton.jit
def fused_cat_sigmoid_sub_mul_kernel(
    a_ptr, b_ptr, c_ptr, out_ptr,
    N1, N2, N3, N_total,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)   # batch
    pid_n = tl.program_id(1)   # inner block

    offsets = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N_total

    in_a = offsets < N1
    in_b = (offsets >= N1) & (offsets < N1 + N2)
    in_c = offsets >= (N1 + N2)

    b_off = tl.maximum(offsets - N1,      0)
    c_off = tl.maximum(offsets - N1 - N2, 0)

    val_a = tl.load(a_ptr + pid_b * N1 + offsets, mask=mask & in_a, other=0.0)
    val_b = tl.load(b_ptr + pid_b * N2 + b_off,   mask=mask & in_b, other=0.0)
    val_c = tl.load(c_ptr + pid_b * N3 + c_off,   mask=mask & in_c, other=0.0)

    val = val_a + val_b + val_c   # mutual exclusion → only one is non-zero

    val_f32 = val.to(tl.float32)
    val_f32 = tl.sigmoid(val_f32)
    val_f32 = val_f32 - 0.25
    val_f32 = val_f32 * 3.141592653589793

    tl.store(out_ptr + pid_b * N_total + offsets, val_f32.to(val.dtype), mask=mask)


@torch.fx.wrap
def fused_cat_sigmoid_sub_mul(a, b, c):
    B       = a.shape[0]
    N1      = a.shape[2]     # 6400
    N2      = b.shape[2]     # 1600
    N3      = c.shape[2]     #  400
    N_total = N1 + N2 + N3  # 8400

    out = torch.empty((B, 1, N_total), dtype=a.dtype, device=a.device)

    # BLOCK_SIZE=8192: for B≥24, grid fits in ≤1 wave on A30 (56 SMs)
    BLOCK_SIZE = 8192
    grid = (B, triton.cdiv(N_total, BLOCK_SIZE))

    fused_cat_sigmoid_sub_mul_kernel[grid](
        a, b, c, out,
        N1, N2, N3, N_total,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_cat_sigmoid_sub_mul