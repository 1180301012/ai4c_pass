import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul.squeeze(1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Fused matmul+squeeze for [B, 1, K] @ [B, K, N] → [B, N]
#
# Design:
#   • Grid  = (B,)  — one CTA per batch element
#   • Each CTA loads the full [K] row of in_0 and the full [K, N] tile of in_1
#     in BLOCK_K-sized chunks, accumulates in fp32, and stores [N] bf16/fp16.
#   • BLOCK_K and BLOCK_N are compile-time constants (no autotune overhead
#     after the first specialisation).
# ---------------------------------------------------------------------------
@triton.jit
def matmul_squeeze_kernel(
    in0_ptr, in1_ptr, out_ptr,
    B, K, N,
    stride_in0_b,                   # = K   for contiguous [B, 1, K]
    stride_in1_b, stride_in1_k,     # strides of [B, K, N]
    stride_out_b,                   # = N   for contiguous [B, N]
    IS_BF16: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b      = tl.program_id(0)
    n_blk  = tl.program_id(1)

    n_off  = n_blk * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_off < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_off  = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_off < K

        # in_0[b, 0, k]  — shape [BLOCK_K]
        x = tl.load(
            in0_ptr + b * stride_in0_b + k_off,
            mask=k_mask, other=0.0,
        ).to(tl.float32)

        # in_1[b, k, n]  — shape [BLOCK_K, BLOCK_N]
        y = tl.load(
            in1_ptr + b * stride_in1_b
                     + k_off[:, None] * stride_in1_k
                     + n_off[None, :],
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum(x[:, None] * y, axis=0)

    if IS_BF16:
        tl.store(out_ptr + b * stride_out_b + n_off,
                 acc.to(tl.bfloat16), mask=n_mask)
    else:
        tl.store(out_ptr + b * stride_out_b + n_off,
                 acc.to(tl.float16),  mask=n_mask)


@torch.fx.wrap
def matmul_squeeze_func(in_0, in_1):
    B  = in_0.shape[0]
    K  = in_0.shape[2]   # 249
    N  = in_1.shape[2]   # 64

    out    = torch.empty((B, N), dtype=in_0.dtype, device=in_0.device)
    is_bf16 = (in_0.dtype == torch.bfloat16)

    # BLOCK_K=256 covers K=249 in a single loop iteration (fastest path).
    # BLOCK_N=64  covers N=64 in a single column tile.
    # Grid = (B, 1) → one CTA handles the entire [K×N] dot product.
    BLOCK_K = 256
    BLOCK_N = 64
    grid = (B, (N + BLOCK_N - 1) // BLOCK_N)

    matmul_squeeze_kernel[grid](
        in_0, in_1, out,
        B, K, N,
        in_0.stride(0),
        in_1.stride(0), in_1.stride(1),
        out.stride(0),
        IS_BF16=is_bf16,
        BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return out


def replacement_func():
    return matmul_squeeze_func