import torch
import triton
import triton.language as tl


# -------------------------------------------------------------------------
# Pattern: elementwise add followed by layer_norm over the last dim (384)
# Matches: in_5 + in_6 -> layer_norm((384,), weight=in_2, bias=in_1, eps=1e-12)
# The result tmp_6 is returned by the model and is the only useful intermediate.
# The slice/linear/tanh that follow use tmp_6 but don't affect the return
# (dead code — we only replace add+layernorm), so they remain untouched.
# -------------------------------------------------------------------------

def pattern(in_5, in_6, in_2, in_1):
    tmp_5 = in_6 + in_5
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (384,), in_2, in_1, 1e-12)
    return tmp_6


def replacement_args(in_5, in_6, in_2, in_1):
    return (in_5, in_6, in_2, in_1)


# -------------------------------------------------------------------------
# Triton kernel: fused elementwise-add + layer-norm
#
# One Triton program = one row (of length N=384).
# Each program loads both input rows, sums them in float32, then computes
# a numerically-stable layer-norm reduction (two passes over the data),
# scales by the per-channel weight, and adds the per-channel bias.
# -------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=32),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32),
    ],
    key=['N'],
)
@triton.jit
def fused_add_layernorm_kernel(
    x1_ptr, x2_ptr,      # first two addends  [num_rows, N]
    weight_ptr, bias_ptr, # affine parameters [N]
    out_ptr,             # output              [num_rows, N]
    N: tl.constexpr,     # normalised dimension (384)
    BLOCK_SIZE: tl.constexpr,
):
    EPS: tl.constexpr = 1e-12

    row_idx = tl.program_id(0)
    row_start = row_idx * N

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # ---- load & add: up cast to float32 immediately -------------------
    x1 = tl.load(x1_ptr + row_start + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + row_start + offsets, mask=mask, other=0.0)
    x_sum_f32 = x1.to(tl.float32) + x2.to(tl.float32)

    # ---- mean (divide by exact N to keep float32) ---------------------
    mean = tl.sum(x_sum_f32, axis=0) / N

    # ---- zero-out masked lanes before computing variance --------------
    x_diff = x_sum_f32 - mean
    x_sq = tl.where(mask, x_diff * x_diff, 0.0)
    var = tl.sum(x_sq, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    # ---- normalise + affine (all float32) -----------------------------
    x_norm = x_diff * rstd

    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    weight_f32 = weight.to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    bias_f32 = bias.to(tl.float32)

    out_f32 = x_norm * weight_f32 + bias_f32

    # ---- store (float32 auto-cast → original dtype) --------------------
    tl.store(out_ptr + row_start + offsets, out_f32, mask=mask)


# -------------------------------------------------------------------------
# Python wrapper  (must be @torch.fx.wrap so FX does not trace into it)
# -------------------------------------------------------------------------

@torch.fx.wrap
def fused_add_layernorm(in_5, in_6, weight, bias):
    """
    Fused add + layer-norm.
    in_5, in_6 : tensors of shape [B, T, N]
    weight, bias: [N]
    returns      : tensor of shape [B, T, N]
    """
    N = in_5.shape[-1]                      # 384
    num_rows = in_5.numel() // N            # B * T  (e.g. 578)

    out = torch.empty_like(in_5)

    # Launch one program per row
    fused_add_layernorm_kernel[(num_rows,)](
        in_5, in_6, weight, bias, out,
        N=N,
    )

    return out


def replacement_func():
    return fused_add_layernorm