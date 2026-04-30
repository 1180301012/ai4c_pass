import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)
    tmp_3 = tmp_2.transpose(-2, -1)
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Fused kernel: layer_norm + transpose(-2,-1) + gelu
#
# Both reads AND writes are coalesced:
#   Load  [BLOCK_K, BLOCK_D]  row-major from input  → coalesced reads
#   After tl.trans → [BLOCK_D, BLOCK_K]
#   Store [BLOCK_D, BLOCK_K]  row-major to output   → coalesced writes
#     (for fixed d, consecutive k values → consecutive addresses).
#
# key=[] means autotune runs once and caches the result for all N=3999 calls.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 32},  num_warps=16, num_stages=1),
        triton.Config({'BLOCK_K': 64},  num_warps=16, num_stages=1),
        triton.Config({'BLOCK_K': 64},  num_warps=32, num_stages=1),
        triton.Config({'BLOCK_K': 128}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_K': 128}, num_warps=32, num_stages=1),
        triton.Config({'BLOCK_K': 256}, num_warps=32, num_stages=1),
        triton.Config({'BLOCK_K': 512}, num_warps=32, num_stages=1),
    ],
    key=[],  # N is always 3999; autotune once, cache forever
)
@triton.jit
def fused_ln_trans_gelu_2d_kernel(
    input_ptr,   # [N, BLOCK_D=512]
    weight_ptr,  # [BLOCK_D=512]
    bias_ptr,    # [BLOCK_D=512]
    output_ptr,  # [BLOCK_D=512, N]  contiguous, row-major
    N,           # runtime number of rows (= 3999)
    eps,
    BLOCK_K: tl.constexpr,   # rows per program (tuned)
    BLOCK_D: tl.constexpr,   # = 512 (full feature dim, power-of-2)
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_K

    rows_k = row_start + tl.arange(0, BLOCK_K)   # [BLOCK_K]
    cols_d = tl.arange(0, BLOCK_D)                # [BLOCK_D]
    row_mask = rows_k < N                          # [BLOCK_K]

    # ---- Load [BLOCK_K, BLOCK_D] (row-major → coalesced reads) ----
    x = tl.load(
        input_ptr + rows_k[:, None] * BLOCK_D + cols_d[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)  # [BLOCK_K, BLOCK_D]

    # ---- Layer Norm (per row, reduce over BLOCK_D) ----
    mean    = tl.sum(x, axis=1) / BLOCK_D             # [BLOCK_K]
    # Inlined diff: frees x registers before variance sum
    var     = tl.sum((x - mean[:, None]) * (x - mean[:, None]), axis=1) / BLOCK_D  # [BLOCK_K]
    inv_std = tl.rsqrt(var + eps)                     # [BLOCK_K]

    weight = tl.load(weight_ptr + cols_d).to(tl.float32)  # [BLOCK_D]
    bias_v = tl.load(bias_ptr   + cols_d).to(tl.float32)  # [BLOCK_D]

    # Inline GELU over normalized (no separate diff / normalized tensors)
    normalized = (x - mean[:, None]) * inv_std[:, None] * weight[None, :] + bias_v[None, :]

    # ---- GELU: exact via erf ----
    sqrt2_inv = 0.7071067811865476
    gelu_out  = normalized * 0.5 * (1.0 + tl.math.erf(normalized * sqrt2_inv))

    # ---- Transpose [BLOCK_K, BLOCK_D] → [BLOCK_D, BLOCK_K] ----
    gelu_T = tl.trans(gelu_out)  # [BLOCK_D, BLOCK_K]

    # ---- Coalesced store [BLOCK_D, BLOCK_K] (k is fast axis) ----
    # ptrs[j, k] = j * N + (row_start + k); for fixed j, k=0..BLOCK_K-1 are
    # consecutive addresses → coalesced writes when BLOCK_K >= 32.
    tl.store(
        output_ptr + cols_d[:, None] * N + rows_k[None, :],
        gelu_T,
        mask=row_mask[None, :],
    )


@torch.fx.wrap
def fused_ln_trans_gelu(in_0, in_1, in_2):
    """
    in_0 : bias   [512]
    in_1 : weight [512]
    in_2 : input  [1, 3999, 512]
    returns : gelu(transpose(layer_norm(in_2)))  shape [1, 512, 3999]
    """
    B, N, D = in_2.shape  # B=1, N=3999, D=512

    # Transposed output [B, D, N]
    output = torch.empty((B, D, N), dtype=in_2.dtype, device=in_2.device)

    # Lambda grid: correct program count for every BLOCK_K
    fused_ln_trans_gelu_2d_kernel[
        lambda meta: ((N + meta['BLOCK_K'] - 1) // meta['BLOCK_K'],)
    ](
        in_2, in_1, in_0, output,
        N, 1e-5,
        BLOCK_D=512,
    )

    return output


def replacement_func():
    return fused_ln_trans_gelu