import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    tmp_3 = tmp_2[(slice(None, None, None), slice(1, None, None))]
    tmp_4 = torch._C._nn.pad(tmp_3, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
    tmp_5 = tmp_2[(slice(None, None, None), slice(None, -1, None))]
    tmp_6 = torch._C._nn.pad(tmp_5, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
    tmp_7 = torch.cat([tmp_4, tmp_2, tmp_6], dim=2)
    return tmp_7


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_trigram_embedding_kernel(
    indices_ptr,
    weight_ptr,
    output_ptr,
    S,
    E: tl.constexpr,
):
    # Each program handles one (batch, seq_position) pair
    pid = tl.program_id(0)
    s = pid % S

    e_off = tl.arange(0, E)

    # Load current index and embedding
    idx_cur = tl.load(indices_ptr + pid)
    cur_emb = tl.load(weight_ptr + idx_cur * E + e_off)

    # Next embedding (position s gets embedding from position s+1)
    # If s+1 >= S, output zeros (boundary condition)
    has_next = (s + 1) < S
    safe_next_pid = tl.where(has_next, pid + 1, pid)
    idx_next = tl.load(indices_ptr + safe_next_pid)
    effective_E_next = tl.where(has_next, E, 0)
    next_mask = e_off < effective_E_next
    next_emb = tl.load(weight_ptr + idx_next * E + e_off, mask=next_mask, other=0.0)

    # Prev embedding (position s gets embedding from position s-1)
    # If s-1 < 0, output zeros (boundary condition)
    has_prev = s > 0
    safe_prev_pid = tl.where(has_prev, pid - 1, pid)
    idx_prev = tl.load(indices_ptr + safe_prev_pid)
    effective_E_prev = tl.where(has_prev, E, 0)
    prev_mask = e_off < effective_E_prev
    prev_emb = tl.load(weight_ptr + idx_prev * E + e_off, mask=prev_mask, other=0.0)

    # Store output: [next_emb | cur_emb | prev_emb] along embedding dim
    # This matches cat([shifted_left, original, shifted_right], dim=2)
    out_base = pid * (3 * E)
    tl.store(output_ptr + out_base + e_off, next_emb)
    tl.store(output_ptr + out_base + E + e_off, cur_emb)
    tl.store(output_ptr + out_base + 2 * E + e_off, prev_emb)


@torch.fx.wrap
def fused_trigram_embedding(in_0, in_1):
    B, S = in_0.shape
    V, E = in_1.shape

    output = torch.empty(B, S, 3 * E, dtype=in_1.dtype, device=in_0.device)

    grid = (B * S,)
    fused_trigram_embedding_kernel[grid](
        in_0, in_1, output,
        S, E,
        num_warps=4,
    )

    return output


def replacement_func():
    return fused_trigram_embedding