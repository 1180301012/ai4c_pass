import torch
import triton
import triton.language as tl


@triton.jit
def _linear_small_n_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    BLOCK_K: tl.constexpr,
):
    """
    (M, N) grid — one program per output element.
    Dot product accumulated in float32; BLOCK_K=64 divides K=448 exactly (no masking).
    Grid: (M, N)
    """
    row = tl.program_id(0)
    col = tl.program_id(1)

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        x_val = tl.load(x_ptr + row * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        w_val = tl.load(w_ptr + col * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        acc += x_val * w_val

    total = tl.sum(acc, axis=0)
    bias_val = tl.load(b_ptr + col).to(tl.float32)
    tl.store(out_ptr + row * N + col, total + bias_val)


@torch.fx.wrap
def triton_linear_small_n(x, weight, bias):
    M = x.shape[0]
    K = x.shape[1]
    N = weight.shape[0]   # always 2

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # BLOCK_K=64: K=448 = 7×64 exactly (zero masking), num_warps=4 — empirically best config
    _linear_small_n_kernel[(M, N)](
        x, weight, bias, out,
        M, N, K,
        BLOCK_K=64, num_warps=4,
    )

    return out


def pattern(x, weight, bias):
    return torch.nn.functional.linear(x, weight, bias)


def replacement_args(x, weight, bias):
    return (x, weight, bias)


def replacement_func():
    return triton_linear_small_n