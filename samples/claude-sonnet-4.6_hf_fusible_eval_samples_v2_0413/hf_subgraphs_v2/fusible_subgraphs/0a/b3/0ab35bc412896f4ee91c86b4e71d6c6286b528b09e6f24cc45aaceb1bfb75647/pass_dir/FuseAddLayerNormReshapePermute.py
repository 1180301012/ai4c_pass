import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches the exact computation in model.py
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3 + in_2
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (128,), in_1, in_0, 1e-05)
    tmp_4 = tmp_3.reshape(1, 2, 2, -1)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.permute(0, 2, 3, 1)
    tmp_8 = tmp_7.reshape(1, -1, 128)
    return (tmp_8,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Triton kernel — fused residual-add + layer_norm
#
# Grid=(1,), single CTA.
# tl.range loop with num_stages=2: Triton prefetches row (r+1)'s x/y loads
# into registers while row r's mean/variance reduction is executing.
# num_warps=1 → 32 threads, 4 elements per thread, pure warp-shuffle.
# w and b are loop-invariant; Triton hoists them to registers.
# ---------------------------------------------------------------------------
@triton.jit
def fused_add_layernorm_kernel(
    x_ptr,                        # in_2  [4 × 128] contiguous
    y_ptr,                        # in_3  [4 × 128] contiguous
    w_ptr,                        # in_1  weight  [128]
    b_ptr,                        # in_0  bias    [128]
    out_ptr,                      # output [4 × 128] contiguous
    N_ROWS: tl.constexpr,         # 4
    N_COLS: tl.constexpr,         # 128
):
    cols  = tl.arange(0, N_COLS)

    # w and b are loop-invariant → hoisted to registers by Triton
    w = tl.load(w_ptr + cols).to(tl.float32)
    b = tl.load(b_ptr + cols).to(tl.float32)

    inv_n = 1.0 / N_COLS

    # static_range: fully unrolled at compile time for best instruction scheduling
    for r in tl.static_range(N_ROWS):
        base = r * N_COLS
        x    = tl.load(x_ptr + base + cols).to(tl.float32)
        y    = tl.load(y_ptr + base + cols).to(tl.float32)
        z    = x + y

        mean  = tl.sum(z,     axis=0) * inv_n
        mean2 = tl.sum(z * z, axis=0) * inv_n
        rstd  = tl.math.rsqrt(mean2 - mean * mean + 1e-5)

        tl.store(out_ptr + base + cols, (z - mean) * rstd * w + b)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_add_layernorm(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [128]
    in_1 : weight [128]
    in_2 : [1, 4, 128]
    in_3 : [1, 4, 128]
    """
    out = torch.empty_like(in_2)

    fused_add_layernorm_kernel[(1,)](
        in_2, in_3, in_1, in_0, out,
        N_ROWS=4,
        N_COLS=128,
        num_warps=1,
        num_stages=1,
    )

    return out


def replacement_func():
    return fused_add_layernorm