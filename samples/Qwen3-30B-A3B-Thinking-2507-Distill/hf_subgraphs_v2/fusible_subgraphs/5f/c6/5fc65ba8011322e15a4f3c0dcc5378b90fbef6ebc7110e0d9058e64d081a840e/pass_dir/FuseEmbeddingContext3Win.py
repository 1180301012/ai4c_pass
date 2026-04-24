import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    tmp_3 = tmp_2[(slice(None, None, None), slice(1, None, None))]
    tmp_4 = torch._C._nn.pad(tmp_3, [0, 0, 0, 1, 0, 0])
    tmp_5 = tmp_2[(slice(None, None, None), slice(None, -1, None))]
    tmp_6 = torch._C._nn.pad(tmp_5, [0, 0, 1, 0, 0, 0])
    tmp_7 = torch.cat([tmp_4, tmp_2, tmp_6], dim = 2)
    return tmp_7


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def _embedding_context3win_kernel(
    indices_ptr,      # flat [B*S] int64 token indices
    embedding_ptr,    # [V, H] embedding weight table
    output_ptr,       # flat [B*S, 3*H] output
    S,                # sequence length
    H: tl.constexpr, # embedding dim (always 128 here)
):
    # One Triton program per (b, s) position
    idx = tl.program_id(0)
    b = idx // S
    s = idx % S

    # Row offsets into the flat output
    out_base = idx * 3 * H

    # Load the current token id
    token_idx = tl.load(indices_ptr + idx)

    # Compute offset vectors for the embedding head dimension
    offsets = tl.arange(0, H)

    # ---- Previous token (s-1) ----
    # Condition: s > 0  =>  s-1 >= 0
    prev_s = s - 1
    prev_valid = s > 0
    # Clamp to valid range to avoid OOB pointer arithmetic when masked
    prev_s_safe = tl.maximum(prev_s, 0)
    prev_tok = tl.load(embedding_ptr + tl.where(prev_valid, prev_safe * H + offsets, offsets),
                       mask=prev_valid, other=0.0)

    # ---- Current token (s) ----
    cur_tok = tl.load(embedding_ptr + token_idx * H + offsets)

    # ---- Next token (s+1) ----
    # Condition: s < S-1  =>  s+1 < S
    next_s = s + 1
    next_valid = s < S - 1
    next_s_safe = tl.minimum(next_s, S - 1)
    next_tok = tl.load(embedding_ptr + tl.where(next_valid, next_s_safe * H + offsets, offsets),
                       mask=next_valid, other=0.0)

    # Store: [prev | cur | next] -> output[idx, 0:H], output[idx, H:2H], output[idx, 2H:3H]
    tl.store(output_ptr + out_base + offsets, prev_tok)
    tl.store(output_ptr + out_base + H + offsets, cur_tok)
    tl.store(output_ptr + out_base + 2 * H + offsets, next_tok)


@torch.fx.wrap
def embedding_context3win(in_0, in_1):
    B = in_0.shape[0]
    S = in_0.shape[1]
    H = in_1.shape[1]   # Always 128 for MobileBERT

    # Output: [B, S, 3*H]
    out = torch.empty((B, S, 3 * H), dtype=in_1.dtype, device=in_1.device)

    _embedding_context3win_kernel[(B * S,)](
        in_0.reshape(-1),   # flat token indices
        in_1,               # embedding weight table
        out.reshape(-1, 3 * H),  # flat [B*S, 3*H] output
        S,
        H,
    )

    return out


def replacement_func():
    return embedding_context3win