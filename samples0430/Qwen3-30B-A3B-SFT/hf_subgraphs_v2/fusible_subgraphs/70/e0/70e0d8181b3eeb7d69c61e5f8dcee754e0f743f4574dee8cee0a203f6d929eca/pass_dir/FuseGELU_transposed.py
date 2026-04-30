import torch
import triton
import triton.language as tl


def pattern(x):
    """Match just the GELU activation applied to the transposed layer_norm output."""
    tmp_4 = torch.nn.functional.gelu(x)
    return tmp_4


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Fused GELU kernel: read [BLOCK_D, BLOCK_K] coalesced, write same layout
# coalesced after tl.trans — NO layer_norm, just element-wise GELU.
# Input shape after layer_norm+transpose: [1, 512, 3999] → treat as [512, 3999].
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 32},  num_warps=8),
        triton.Config({'BLOCK_K': 32},  num_warps=16),
        triton.Config({'BLOCK_K': 64},  num_warps=16),
        triton.Config({'BLOCK_K': 64},  num_warps=32),
        triton.Config({'BLOCK_K': 128}, num_warps=16),
        triton.Config({'BLOCK_K': 128}, num_warps=32),
    ],
    key=[],
)
@triton.jit
def gelu_transposed_kernel(
    input_ptr,
    output_ptr,
    N,              # = 3999 (sequence length)
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,  # = 512
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_K

    rows_k = row_start + tl.arange(0, BLOCK_K)   # [BLOCK_K]
    cols_d = tl.arange(0, BLOCK_D)                # [BLOCK_D]
    row_mask = rows_k < N

    # Load [BLOCK_K, BLOCK_D] row-major → coalesced reads
    x = tl.load(
        input_ptr + rows_k[:, None] * BLOCK_D + cols_d[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )  # [BLOCK_K, BLOCK_D]  already float16/bfloat16

    # GELU (exact via erf, in same dtype as input)
    x_f32  = x.to(tl.float32)
    sqrt2_inv = 0.7071067811865476
    gelu_f32 = x_f32 * 0.5 * (1.0 + tl.math.erf(x_f32 * sqrt2_inv))
    gelu_out = gelu_f32.to(x.dtype)  # cast back to input dtype

    # Transpose [BLOCK_K, BLOCK_D] → [BLOCK_D, BLOCK_K]
    gelu_T = tl.trans(gelu_out)  # [BLOCK_D, BLOCK_K]

    # Coalesced store [BLOCK_D, BLOCK_K]: ptrs[j,k] = j*N + (row_start+k)
    tl.store(
        output_ptr + cols_d[:, None] * N + rows_k[None, :],
        gelu_T,
        mask=row_mask[None, :],
    )


@torch.fx.wrap
def triton_gelu_transposed(x):
    """
    x : [1, 512, 3999]  (= layer_norm output already transposed)
    returns gelu(x) with same shape [1, 512, 3999]
    """
    B, D, N = x.shape  # B=1, D=512, N=3999
    output = torch.empty((B, D, N), dtype=x.dtype, device=x.device)

    gelu_transposed_kernel[
        lambda meta: ((N + meta['BLOCK_K'] - 1) // meta['BLOCK_K'],)
    ](
        x, output, N,
        BLOCK_D=512,
    )

    return output


def replacement_func():
    return triton_gelu_transposed