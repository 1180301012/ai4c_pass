import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fused unsqueeze(1) + transpose(2, 3)
#
# Input  shape: [B, K, J]  (B=1, K=1024, J=128)  — contiguous
# Output shape: [B, 1, J, K]  — contiguous
#
# out[b, 0, j, k] = in[b, k, j]
#
# 2-D tiling: load [BLOCK_K, BLOCK_J] (coalesced reads, j fastest),
# transpose in registers, store [BLOCK_J, BLOCK_K] (coalesced writes, k fastest).
# J%BLOCK_J==0 and K%BLOCK_K==0, so no boundary masking needed.
# ---------------------------------------------------------------------------

@triton.jit
def _transpose_kernel(
    in_ptr,
    out_ptr,
    J,
    K,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_j = tl.program_id(0)
    pid_k = tl.program_id(1)

    j_start = pid_j * BLOCK_J
    k_start = pid_k * BLOCK_K

    j_offs = j_start + tl.arange(0, BLOCK_J)   # [BLOCK_J]
    k_offs = k_start + tl.arange(0, BLOCK_K)   # [BLOCK_K]

    # ---------------------------------------------------------------
    # Load [BLOCK_K, BLOCK_J] from input [K, J]
    # in[k, j] → k*J + j
    # Warp reads BLOCK_J consecutive j-values per k → coalesced reads
    # ---------------------------------------------------------------
    in_offs = k_offs[:, None] * J + j_offs[None, :]   # [BLOCK_K, BLOCK_J]
    x = tl.load(in_ptr + in_offs)   # no mask: J%BLOCK_J == 0

    # ---------------------------------------------------------------
    # Store [BLOCK_J, BLOCK_K] to output [J, K]
    # out[j, k] → j*K + k
    # Warp writes BLOCK_K consecutive k-values per j → coalesced writes
    # ---------------------------------------------------------------
    out_offs = j_offs[:, None] * K + k_offs[None, :]  # [BLOCK_J, BLOCK_K]
    tl.store(out_ptr + out_offs, tl.trans(x))


@torch.fx.wrap
def fused_unsqueeze_transpose_1_1024_128(x):
    B = x.shape[0]   # 1
    K = x.shape[1]   # 1024
    J = x.shape[2]   # 128

    out = torch.empty((B, 1, J, K), dtype=x.dtype, device=x.device)

    # BLOCK_J=128, BLOCK_K=32: grid=(1,32)=32 blocks — minimal scheduling overhead,
    # all blocks fit in one GPU wave; exact divisors → no masking needed
    BLOCK_J = 128
    BLOCK_K = 32
    grid = (triton.cdiv(J, BLOCK_J), triton.cdiv(K, BLOCK_K))
    _transpose_kernel[grid](x, out, J, K, BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K,
                            num_warps=4)
    return out


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

def pattern(x):
    tmp_1 = x.unsqueeze(1)
    tmp_2 = tmp_1.transpose(2, 3)
    return tmp_2


def replacement_args(x):
    return (x,)


def replacement_func():
    return fused_unsqueeze_transpose_1_1024_128