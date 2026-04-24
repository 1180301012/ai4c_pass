import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 16}),
        triton.Config({'BLOCK_N': 32}),
        triton.Config({'BLOCK_N': 64}),
    ],
    key=['N'],
)
@triton.jit
def _causal_attn_mask_kernel(
    in_0_ptr,
    out_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    """
    Compute the causal attention mask.

    For each query position q and key position k (0-indexed, k is cols):
      - if k > q  →  masked by causality  →  output = -3.4028234663852886e+38
      - if in_0[0, k] == 0  →  masked by input  →  output = -3.4028234663852886e+38
      - otherwise  →  output = 0.0

    Grid: (N,)  — each program owns one query row of length N.
    """
    q = tl.program_id(0)                     # query position [0, N)
    k = tl.arange(0, BLOCK_N)                # key positions [0, BLOCK_N)
    mask = k < N                             # valid key positions

    # Load in_0[0, k]; out-of-bounds → 0  (represents "masked")
    in_0_val = tl.load(in_0_ptr + k, mask=mask, other=0)

    # Causal condition: output -3.4e38 if (k > q) OR (in_0 is 0)
    vals = tl.where((k > q) | (in_0_val == 0),
                    -3.4028234663852886e+38,
                    0.0)

    tl.store(out_ptr + q * N + k, vals.to(tl.float32), mask=mask)


@torch.fx.wrap
def causal_attention_mask(in_0):
    """
    Drop-in replacement for the causal attention mask construction.

    Args:
        in_0: int64 tensor of shape (1, N) — attention mask (1 = unmasked).

    Returns:
        Float32 tensor of shape (1, 1, N, N) — causal + input mask.
    """
    N = in_0.shape[1]
    out = torch.empty((1, 1, N, N), dtype=torch.float32, device=in_0.device)
    _causal_attn_mask_kernel[(N,)](in_0, out, N)
    return out