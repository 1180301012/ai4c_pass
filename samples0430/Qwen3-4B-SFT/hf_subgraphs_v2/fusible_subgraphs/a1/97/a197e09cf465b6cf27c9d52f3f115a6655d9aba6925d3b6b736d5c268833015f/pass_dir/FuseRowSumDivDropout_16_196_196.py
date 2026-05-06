import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – mirrors model.py exactly (no None/cleanup lines)
# ---------------------------------------------------------------------------
def pattern(x):
    tmp_0 = x.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    x /= tmp_1
    tmp_3 = torch.nn.functional.dropout(x, 0.0, False, False)
    return tmp_3


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel: row-sum normalization (sum → add_dim → div → store)
#
# Each Triton program handles ONE row of the [B*K, N] view.
# Accumulates in fp32 for precision; stores back in the input dtype.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_N': 256}, num_warps=16),
        triton.Config({'BLOCK_N': 512}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 512}, num_warps=16),
    ],
    key=['N', 'K'],
)
@triton.jit
def row_sum_div_kernel(
    input_ptr,
    output_ptr,
    K,
    N,
    stride_bk,
    BLOCK_N: tl.constexpr,
):
    # One program per row
    row_idx = tl.program_id(0)
    batch_idx = row_idx // K
    key_idx   = row_idx % K

    offsets = tl.arange(0, BLOCK_N)
    mask    = offsets < N

    # Accumulate sum in fp32 for numerical stability
    s = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k in range(K):
        base = batch_idx * (K * N) + key_idx * N + k * stride_bk
        x_v  = tl.load(input_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        s   += x_v

    inv_N = tl.sum(s, axis=0) / N          # scalar: total sum / N

    for k in range(K):
        base = batch_idx * (K * N) + key_idx * N + k * stride_bk
        x_v  = tl.load(input_ptr + base + offsets, mask=mask, other=0.0)
        out_v = (x_v.to(tl.float32) * inv_N).to(x_v.dtype)
        tl.store(output_ptr + base + offsets, out_v, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def row_sum_div_triton(x):
    B, K, N = x.shape[0], x.shape[1], x.shape[2]
    rows = B * K
    output = torch.empty_like(x)

    row_sum_div_kernel[(rows,)](
        x, output,
        K, N,
        N,          # stride_bk = N for contiguous tensor
    )
    return output


# ---------------------------------------------------------------------------
# replacement_func – returns the wrapper (NOT a call)
# ---------------------------------------------------------------------------
def replacement_func():
    return row_sum_div_triton