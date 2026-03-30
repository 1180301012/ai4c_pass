import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match x.mean(dim=-2, keepdim=True) for any 3-D input tensor
# ---------------------------------------------------------------------------
def pattern(x):
    return x.mean(dim=-2, keepdim=True)


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel – reduces along dim=1 of a (B, M, N) tensor.
# 1-D grid: one program per batch item (pid_0 = batch index).
# ALL N channels processed in a single program (BLOCK_N = N = 256).
# Accumulation always in float32 for precision.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        # 7 configs spanning the full BLOCK_M range.
        # BLOCK_M=256 is important for B=12/16 (small batch, fits memory pipeline).
        # BLOCK_M=512/1024 help large B cases.  Including num_warps=16 variants
        # enables Triton to explore higher thread-level parallelism when beneficial.
        triton.Config({"BLOCK_M": 128},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_M": 256},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_M": 256},  num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_M": 512},  num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_M": 512},  num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 1024}, num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_M": 1024}, num_warps=16, num_stages=2),
    ],
    key=["B", "M", "N"],
)
@triton.jit
def _mean_dim1_1d_kernel(
    x_ptr,           # input  (B, M, N) – any fp dtype, contiguous
    out_ptr,         # output (B, N)    – float32
    B, M, N,
    BLOCK_M: tl.constexpr,  # tile size along the reduction (M) dimension
    BLOCK_N: tl.constexpr,  # = N; covers all channels in one program
):
    batch_id = tl.program_id(0)
    n_off    = tl.arange(0, BLOCK_N)   # shape (N,) – all channels

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Tile over the reduction (M) dimension
    for m_start in range(0, M, BLOCK_M):
        m_off  = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_off < M

        # x[batch_id, m_off[i], n_off[j]]
        ptrs  = x_ptr + batch_id * M * N + m_off[:, None] * N + n_off[None, :]
        x_val = tl.load(ptrs, mask=m_mask[:, None], other=0.0)

        acc += tl.sum(x_val.to(tl.float32), axis=0)

    acc /= M

    out_ptrs = out_ptr + batch_id * N + n_off
    tl.store(out_ptrs, acc)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_mean_dim_m2_keepdim(x):
    """Drop-in replacement for x.mean(dim=-2, keepdim=True) on 3-D tensors."""
    B = x.shape[0]
    M = x.shape[1]
    N = x.shape[2]

    # For very small batch sizes Triton launch overhead outweighs the benefit;
    # fall back to PyTorch's optimised CUDA implementation.
    if B <= 4:
        return x.mean(dim=-2, keepdim=True)

    # Float32 intermediate; avoids precision loss for fp16/bf16 inputs.
    out_f32 = torch.empty((B, N), dtype=torch.float32, device=x.device)

    # BLOCK_N is a compile-time constant; next_power_of_2(256) = 256.
    BLOCK_N = triton.next_power_of_2(N)

    # 1-D grid: one CTA per batch element.
    _mean_dim1_1d_kernel[(B,)](
        x, out_f32,
        B, M, N,
        BLOCK_N=BLOCK_N,
    )

    # Cast back to the original dtype and restore the keepdim axis.
    return out_f32.to(x.dtype).view(B, 1, N)


def replacement_func():
    return triton_mean_dim_m2_keepdim