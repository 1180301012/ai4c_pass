import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match torch.nn.functional.embedding with the exact call signature
# ---------------------------------------------------------------------------
def pattern(in_1, in_2):
    return torch.nn.functional.embedding(in_1, in_2, None, None, 2.0, False, False)


def replacement_args(in_1, in_2):
    return (in_1, in_2)


# ---------------------------------------------------------------------------
# Triton embedding kernel – fixed config, NO autotune
#
# Autotune was removed because occasional autotune-cache lookups during the
# 100 measured trials cause timing spikes (IQR/median > 20%) that trigger
# "environment fluctuation" failures.  A fixed single-config kernel has:
#   • no cache lookup overhead
#   • fully deterministic dispatch path after the first JIT compile
#   • more stable timing under GPU frequency/thermal variation
#
# Fixed config: BLOCK_D=1024, num_warps=4 (128 threads)
#   elements/thread = 1024 / 128 = 8 bfloat16 = 16 bytes = 128-bit load ✓
#   iterations      = ceil(1536/1024) = 2
#     iter 0: D=[0..1023]  – no masking
#     iter 1: D=[1024..2047], only [1024..1535] valid – 512 elements used
#   coalesced reads: 128 threads × 8 bf16 = 1024 consecutive bfloat16 per iter
# ---------------------------------------------------------------------------
@triton.jit
def embedding_forward_kernel(
    indices_ptr,            # int64  [n_tokens]    – vocab indices (flat)
    weight_ptr,             # dtype  [V, D]        – embedding table
    output_ptr,             # dtype  [n_tokens, D] – output (flat)
    D,                      # embedding dimension  (runtime)
    BLOCK_D: tl.constexpr,
):
    token_id = tl.program_id(0)

    # Load the vocab index for this token (scalar broadcast)
    idx = tl.load(indices_ptr + token_id).to(tl.int64)

    weight_row_base = idx * D
    output_row_base = token_id * D

    d_off   = tl.arange(0, BLOCK_D)
    n_iters = (D + BLOCK_D - 1) // BLOCK_D

    for k in range(n_iters):
        cur_d = k * BLOCK_D + d_off
        mask  = cur_d < D
        val = tl.load(weight_ptr + weight_row_base + cur_d,
                      mask=mask, other=0.0)
        tl.store(output_ptr + output_row_base + cur_d, val, mask=mask)


# ---------------------------------------------------------------------------
# Kernel wrapper (must be @torch.fx.wrap so FX doesn't trace inside)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_embedding_lookup(indices, weight):
    D        = weight.shape[1]      # 1536 for jina-reranker-m0
    n_tokens = indices.numel()
    idx_shape = indices.shape       # e.g. (128, 64)

    output = torch.empty((*idx_shape, D), dtype=weight.dtype, device=weight.device)

    # Fixed config: BLOCK_D=1024, num_warps=4
    #   • 128 threads → 8 bf16/thread = 128-bit loads (optimal bandwidth)
    #   • 2 loop iterations: [0..1023] fully used + [1024..1535] used/masked
    #   • No autotune: one JIT compilation, cached for all subsequent calls
    embedding_forward_kernel[(n_tokens,)](
        indices,
        weight,
        output,
        D,
        BLOCK_D=1024,
        num_warps=4,
    )

    return output


# ---------------------------------------------------------------------------
# Replacement entry point
# ---------------------------------------------------------------------------
def replacement_func():
    return triton_embedding_lookup