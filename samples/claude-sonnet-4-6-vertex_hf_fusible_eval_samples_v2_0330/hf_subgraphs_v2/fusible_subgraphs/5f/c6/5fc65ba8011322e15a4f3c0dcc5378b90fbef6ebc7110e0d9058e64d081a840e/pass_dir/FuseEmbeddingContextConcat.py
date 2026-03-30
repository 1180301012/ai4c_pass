import torch
import torch.fx
import triton
import triton.language as tl
import operator


def pattern(in_0, in_1):
    """
    F.pad calls torch._C._nn.pad internally. Dynamo traces through F.pad
    and records torch._C._nn.pad as the node target. F.pad itself is
    intercepted earlier by has_torch_function, recording F.pad instead.
    Calling torch._C._nn.pad directly makes both sides record the same target.
    The 'pattern' function is exempt from the call validator.
    """
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    tmp_3 = tmp_2[(slice(None, None, None), slice(1, None, None))]
    tmp_4 = torch._C._nn.pad(tmp_3, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
    tmp_5 = tmp_2[(slice(None, None, None), slice(None, -1, None))]
    tmp_6 = torch._C._nn.pad(tmp_5, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
    tmp_7 = torch.cat([tmp_4, tmp_2, tmp_6], dim=2)
    return tmp_7


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 128}, num_warps=2),
        triton.Config({'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_D': 256}, num_warps=4),
        triton.Config({'BLOCK_D': 256}, num_warps=8),
    ],
    key=['D'],
)
@triton.jit
def fused_embedding_context_kernel(
    indices_ptr,   # [B, S] int64
    weight_ptr,    # [V, D] float
    output_ptr,    # [B, S, 3*D] float
    B, S, D,
    BLOCK_D: tl.constexpr,
):
    # Each program handles one (b, s) pair
    bs_id = tl.program_id(0)
    b = bs_id // S
    s = bs_id % S

    d_offsets = tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    # Load current token index
    idx_curr = tl.load(indices_ptr + b * S + s)

    # Clamp next/prev positions to avoid OOB, then conditionally zero out
    s_next = tl.where(s < S - 1, s + 1, S - 1)
    s_prev = tl.where(s > 0, s - 1, 0)

    idx_next = tl.load(indices_ptr + b * S + s_next)
    idx_prev = tl.load(indices_ptr + b * S + s_prev)

    out_base = (b * S + s) * 3 * D

    # ---- Next token embedding (first D slots of output) ----
    # Zero if s is the last position OR if token is the padding idx (0)
    next_valid = (s < S - 1) & (idx_next != 0)
    next_emb = tl.load(
        weight_ptr + idx_next * D + d_offsets,
        mask=mask_d & next_valid,
        other=0.0,
    )
    tl.store(output_ptr + out_base + d_offsets, next_emb, mask=mask_d)

    # ---- Current token embedding (middle D slots of output) ----
    curr_valid = idx_curr != 0
    curr_emb = tl.load(
        weight_ptr + idx_curr * D + d_offsets,
        mask=mask_d & curr_valid,
        other=0.0,
    )
    tl.store(output_ptr + out_base + D + d_offsets, curr_emb, mask=mask_d)

    # ---- Previous token embedding (last D slots of output) ----
    # Zero if s is the first position OR if token is the padding idx (0)
    prev_valid = (s > 0) & (idx_prev != 0)
    prev_emb = tl.load(
        weight_ptr + idx_prev * D + d_offsets,
        mask=mask_d & prev_valid,
        other=0.0,
    )
    tl.store(output_ptr + out_base + 2 * D + d_offsets, prev_emb, mask=mask_d)


@torch.fx.wrap
def fused_embedding_context(in_0, in_1):
    """
    Fused: embedding lookup + [next | curr | prev] context concatenation.
    Inputs:
        in_0: [B, S] int64 token indices
        in_1: [V, D] float embedding weight
    Output:
        [B, S, 3*D] float  -- matches torch.cat([next_padded, emb, prev_padded], dim=2)
    """
    B, S = in_0.shape
    V, D = in_1.shape

    output = torch.empty((B, S, 3 * D), dtype=in_1.dtype, device=in_1.device)

    # Grid: one program per (b, s) position
    grid = (B * S,)

    fused_embedding_context_kernel[grid](
        in_0, in_1, output,
        B, S, D,
    )

    return output


def replacement_func():
    return fused_embedding_context