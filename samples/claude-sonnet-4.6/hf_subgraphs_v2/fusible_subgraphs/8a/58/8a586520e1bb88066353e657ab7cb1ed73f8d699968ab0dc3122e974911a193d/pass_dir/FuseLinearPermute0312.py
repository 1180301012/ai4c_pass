import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.permute(0, 3, 1, 2)
    return tmp_3


def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)


# ---------------------------------------------------------------------------
# Grid: (B*K,  ceil(IJ / BLOCK_IJ))
#
# One CTA per (b, k) pair.  For each IJ tile the kernel:
#   1. Loads  x[b, ij_tile, 0:M]   (M=3 constexpr, tl.static_range → unrolled)
#   2. Loads  scalar w[k, 0:M] and bias[k]
#   3. Accumulates in fp32
#   4. Stores BLOCK_IJ contiguous values to out[b, k, ij_tile]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_IJ': 128},  num_warps=4),
        triton.Config({'BLOCK_IJ': 256},  num_warps=4),
        triton.Config({'BLOCK_IJ': 512},  num_warps=8),
        triton.Config({'BLOCK_IJ': 1024}, num_warps=8),
    ],
    key=['B', 'K', 'IJ'],
    reset_to_zero=[],
)
@triton.jit
def _linear_permute_kernel(
    x_ptr,    # [B, IJ, M]  input  (contiguous, M=3)
    w_ptr,    # [K, M]      weight
    b_ptr,    # [K]         bias
    out_ptr,  # [B, K, IJ] output  (permuted; stride-1 within each k-slice)
    B, K, IJ,
    M: tl.constexpr,
    BLOCK_IJ: tl.constexpr,
):
    pid_bk = tl.program_id(0)
    pid_ij = tl.program_id(1)

    b_idx = pid_bk // K
    k_idx = pid_bk % K

    ij_start = pid_ij * BLOCK_IJ
    ij_offs  = ij_start + tl.arange(0, BLOCK_IJ)
    ij_mask  = ij_offs < IJ

    x_base = b_idx * IJ * M
    acc    = tl.zeros([BLOCK_IJ], dtype=tl.float32)

    for m in tl.static_range(M):
        xv = tl.load(x_ptr + x_base + ij_offs * M + m,
                     mask=ij_mask, other=0.0)
        wv = tl.load(w_ptr + k_idx * M + m)
        acc += xv.to(tl.float32) * wv.to(tl.float32)

    bias_k = tl.load(b_ptr + k_idx)
    acc   += bias_k.to(tl.float32)

    out_base = pid_bk * IJ
    tl.store(out_ptr + out_base + ij_offs, acc, mask=ij_mask)


@torch.fx.wrap
def fused_linear_permute(in_0, in_1, in_3):
    """
    Fused replacement for:
        linear = F.linear(in_3, in_1, in_0)  # [B,I,J,M] -> [B,I,J,K]
        out    = linear.permute(0, 3, 1, 2)  # -> [B,K,I,J]
    """
    B, I, J, M = in_3.shape
    K  = in_1.shape[0]
    IJ = I * J

    device   = in_3.device
    in_0_dev = in_0.to(device)
    in_1_dev = in_1.to(device)
    in_3_c   = in_3.contiguous()

    out = torch.empty((B, K, I, J), dtype=in_3.dtype, device=device)

    grid = lambda meta: (B * K, triton.cdiv(IJ, meta['BLOCK_IJ']))

    _linear_permute_kernel[grid](
        in_3_c, in_1_dev, in_0_dev, out,
        B, K, IJ, M,
    )

    return out


def replacement_func():
    return fused_linear_permute