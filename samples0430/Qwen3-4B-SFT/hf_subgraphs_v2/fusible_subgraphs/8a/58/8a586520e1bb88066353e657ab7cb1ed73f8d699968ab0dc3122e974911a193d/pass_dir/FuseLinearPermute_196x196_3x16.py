import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────
#  Pattern: torch.nn.functional.linear(in_3, in_1, in_0)
#          in_0 = bias   [16]
#          in_1 = weight [16, 3]
#          in_3 = x      [1, 196, 196, 3]
#          output       [1, 196, 196, 16]
# ─────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_3):
    return torch.nn.functional.linear(in_3, in_1, in_0)


def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)


# ─────────────────────────────────────────────────────────────
#  Triton kernel – fully unrolled, fp32 accumulation
#  Grid: (ceil(M / BLOCK_M),)    M = B*S*S2 = 38416
# ─────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64}),
        triton.Config({'BLOCK_M': 128}),
        triton.Config({'BLOCK_M': 256}),
        triton.Config({'BLOCK_M': 512}),
        triton.Config({'BLOCK_M': 1024}),
        triton.Config({'BLOCK_M': 2048}),
    ],
    key=['M'],
)
@triton.jit
def _linear_kernel(
    x_ptr,    # [M, K]
    w_ptr,    # [N, K]
    b_ptr,    # [N]
    out_ptr,  # [M, N]
    M,
    K: tl.constexpr,    # 3
    N: tl.constexpr,    # 16
    BLOCK_M: tl.constexpr,
):
    pid   = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    for n in tl.static_range(N):
        acc = tl.zeros([BLOCK_M], dtype=tl.float32)
        for k in tl.static_range(K):
            x_val = tl.load(x_ptr + offs_m * K + k, mask=mask_m, other=0.0)
            w_val = tl.load(w_ptr + n * K + k)
            acc  += x_val.to(tl.float32) * w_val.to(tl.float32)
        b_val = tl.load(b_ptr + n)
        acc  += b_val.to(tl.float32)
        tl.store(out_ptr + offs_m * N + n, acc.to(x_ptr.dtype.element_ty), mask=mask_m)


@torch.fx.wrap
def linear_opt(in_0, in_1, in_3):
    B   = in_3.shape[0]
    S   = in_3.shape[1]
    S2  = in_3.shape[2]
    K   = in_3.shape[3]
    N   = in_1.shape[0]
    M   = B * S * S2
    out = torch.empty((B, S, S2, N), dtype=in_3.dtype, device=in_3.device)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
    _linear_kernel[grid](in_3, in_1, in_0, out, M, K=K, N=N)
    return out


def replacement_func():
    return linear_opt