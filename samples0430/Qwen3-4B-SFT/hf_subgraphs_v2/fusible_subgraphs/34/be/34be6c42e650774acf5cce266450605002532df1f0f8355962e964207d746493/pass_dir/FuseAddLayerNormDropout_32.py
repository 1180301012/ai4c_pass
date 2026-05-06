import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
    ],
    key=['hidden_size'],
)
@triton.jit
def fused_add_layernorm_kernel_32(
    word_emb_ptr,
    pos_emb_ptr,
    out_ptr,
    LN_weight_ptr,
    LN_bias_ptr,
    eps,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: computes (word_emb + pos_emb), then layer-norm, then dropout (no-op in eval).
    Each program handles one row (one sequence position). hidden_size = 32.
    """
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size

    # Load both embeddings and sum; convert to float32 for numerical stability
    row_offset = row_idx * hidden_size
    word_emb = tl.load(word_emb_ptr + row_offset + offsets, mask=mask, other=0.0)
    pos_emb  = tl.load(pos_emb_ptr  + row_offset + offsets, mask=mask, other=0.0)
    x = word_emb + pos_emb

    # Layer norm: compute mean
    x_sum = tl.sum(x.to(tl.float32), axis=0)
    mean = x_sum / hidden_size

    # Layer norm: compute variance
    x_centered = x - mean
    x_centered_sq = tl.where(mask, x_centered * x_centered, 0.0)
    var = tl.sum(x_centered_sq, axis=0) / hidden_size
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize
    x_norm = x_centered * rstd

    # Load LN weight and bias
    w = tl.load(LN_weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(LN_bias_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)

    # Scale and shift
    result = x_norm * w + b

    # Store as original dtype (float16 or bfloat16)
    tl.store(out_ptr + row_offset + offsets, result, mask=mask)


def pattern(word_emb, pos_emb, ln_weight, ln_bias):
    added = word_emb + pos_emb
    normed = torch.nn.functional.layer_norm(added, (32,), ln_weight, ln_bias, 1e-05)
    dropped = torch.nn.functional.dropout(normed, 0.1, False, False)
    return dropped


def replacement_args(word_emb, pos_emb, ln_weight, ln_bias):
    return (word_emb, pos_emb, ln_weight, ln_bias)


@torch.fx.wrap
def fused_add_layernorm_32(word_emb, pos_emb, ln_weight, ln_bias):
    S = word_emb.shape[0]
    H = word_emb.shape[1]
    N = S * H
    eps = 1e-05

    out = torch.empty_like(word_emb)

    # Grid = (N * num_row_chunks,) processes ceil(H/BLOCK_SIZE) rows per program
    # Use larger block to improve occupancy
    fused_add_layernorm_kernel_32[(N,)](
        word_emb, pos_emb, out, ln_weight, ln_bias,
        eps=eps,
        hidden_size=H,
    )

    return out


def replacement_func():
    return fused_add_layernorm_32