import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_3 = in_3 + in_2
    tmp_4 = tmp_3.float()
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    tmp_9 = tmp_4 - tmp_5
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    tmp_12 = tmp_9 / tmp_11
    tmp_13 = tmp_12.to(torch.float32)
    tmp_14 = in_1 * tmp_13
    tmp_15 = tmp_14 + in_0
    return tmp_15


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def _fused_add_layer_norm_kernel(
    out_ptr,
    in3_ptr,
    in2_ptr,
    in1_ptr,
    in0_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load and add hidden states + residual (cast to fp32 in registers)
    # Masked-out lanes get other=0.0, so they don't pollute either reduction
    in3 = tl.load(in3_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    in2 = tl.load(in2_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    x = in3 + in2

    # Parallel-moments: compute E[x] and E[x²] in one conceptual pass.
    # Masked lanes have x=0 → contribute 0 to both sums → no tl.where needed.
    mean  = tl.sum(x,     axis=0) / N          # E[x]
    mean2 = tl.sum(x * x, axis=0) / N          # E[x²]
    var   = mean2 - mean * mean                 # Var = E[x²] - E[x]²

    # Normalize (masked lanes produce (0-mean)*rstd but are not stored)
    rstd   = tl.math.rsqrt(var + eps)
    x_norm = (x - mean) * rstd

    # Affine transform (weight + bias cast to fp32)
    w   = tl.load(in1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b   = tl.load(in0_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = w * x_norm + b

    tl.store(out_ptr + row * N + cols, out, mask=mask)


@torch.fx.wrap
def fused_add_layer_norm(in_0, in_1, in_2, in_3):
    shape = in_3.shape
    N = shape[-1]
    M = in_3.numel() // N

    in2_c = in_2.contiguous()
    in3_c = in_3.contiguous()

    out = torch.empty(list(shape), dtype=torch.float32, device=in_3.device)

    _fused_add_layer_norm_kernel[(M,)](
        out,
        in3_c,
        in2_c,
        in_1,
        in_0,
        N,
        1e-7,
    )

    return out


def replacement_func():
    return fused_add_layer_norm