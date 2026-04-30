import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # BLOCK_C=64, BLOCK_N=64: small blocks, many programs → high GPU util
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 64}, num_warps=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 64}, num_warps=8),
        # BLOCK_C=128: fewer programs, more work per block
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 128}, num_warps=8),
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 128}, num_warps=16),
        # BLOCK_C=512: all C at once, fewest blocks
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 512}, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 512}, num_warps=16),
    ],
    key=['B', 'N', 'C'],
)
@triton.jit
def _mean_neg2_coalesced_kernel(
    input_ptr,
    output_ptr,
    B,
    N,
    C,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # 2D grid: pid_b in dim-0, pid_c in dim-1
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    n_offs = tl.arange(0, BLOCK_N)   # [BLOCK_N]
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)  # [BLOCK_C]

    n_mask = n_offs < N   # [BLOCK_N]
    c_mask = c_offs < C   # [BLOCK_C]

    # Load [BLOCK_N, BLOCK_C]: inner dim = BLOCK_C (stride 1 in C) → COALESCED
    ptrs = (input_ptr
            + pid_b * N * C
            + n_offs[:, None] * C
            + c_offs[None, :])
    x = tl.load(ptrs,
                mask=n_mask[:, None] & c_mask[None, :],
                other=0.0).to(tl.float32)   # [BLOCK_N, BLOCK_C]

    # Sum along N (axis 0) → [BLOCK_C]
    sums = tl.sum(x, axis=0)
    means = (sums / N.to(tl.float32)).to(x.dtype)

    # Store output[pid_b, c_offs]: inner dim = 1 → COALESCED
    out_ptrs = output_ptr + pid_b * C + c_offs
    tl.store(out_ptrs, means, mask=c_mask)


def _run_mean(x):
    B, N, C = x.shape
    out = torch.empty((B, C), dtype=x.dtype, device=x.device)
    # Grid: (B, c_blocks) — pid_b=block over B, pid_c=block over C
    grid = lambda meta: (B, triton.cdiv(C, meta['BLOCK_C']))
    _mean_neg2_coalesced_kernel[grid](x, out, B, N, C)
    return out


def _run_linear(input, weight, bias):
    B, K = input.shape
    N_out = weight.shape[0]
    out = torch.empty((B, N_out), dtype=input.dtype, device=input.device)
    BLOCK_M = min(32, B)
    BLOCK_N = min(32, N_out)
    grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(N_out, BLOCK_N))
    _linear_bias_kernel[grid](
        input, weight, bias, out,
        B, N_out, K,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_K=min(64, K),
    )
    return out


@torch.fx.wrap
def dispatch_fn(arg0, arg1, route):
    if route == "mean":
        return _run_mean(arg0)
    elif route == "linear":
        return _run_linear(arg0, arg1, arg2)
    return None


def pattern(x):
    return x.mean(-2)


def replacement_args(x):
    return (x, None, "mean")


def replacement_func():
    return dispatch_fn