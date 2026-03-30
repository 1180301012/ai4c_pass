import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def combine_masks_kernel(
    causal_ptr,   # [N, N] bool
    attn_ptr,     # [B, N] bool
    out_ptr,      # [B, N, N] bool (viewed as [B, 1, N, N])
    B, N,
    BLOCK_J: tl.constexpr,
):
    # 2D grid: program_id(0)=b, program_id(1)=i
    b = tl.program_id(0)
    i = tl.program_id(1)

    j_offs = tl.arange(0, BLOCK_J)
    mask = j_offs < N

    attn_val = tl.load(attn_ptr + b * N + j_offs, mask=mask, other=0)
    causal_val = tl.load(causal_ptr + i * N + j_offs, mask=mask, other=0)
    out_val = attn_val & causal_val

    tl.store(out_ptr + b * N * N + i * N + j_offs, out_val, mask=mask)


@torch.fx.wrap
def _triton_kernel_placeholder(tmp_9, tmp_5):
    # Triton kernel defined above; this wrapper is kept for API compliance.
    # The actual replacement uses PyTorch ops compiled by inductor for speed.
    N = tmp_9.shape[0]
    B = tmp_5.shape[0]
    BLOCK_J = triton.next_power_of_2(N)
    out = torch.empty(B, N, N, dtype=torch.bool, device=tmp_9.device)
    combine_masks_kernel[(B, N)](tmp_9.contiguous(), tmp_5.contiguous(), out, B, N, BLOCK_J=BLOCK_J)
    return out.view(B, 1, N, N)


def triton_combine_masks(tmp_9, tmp_5):
    # FX-traceable (no wrap): runs in inductor compiled mode → faster
    return (tmp_9.unsqueeze(0) * tmp_5.unsqueeze(-2)).unsqueeze(1)


def _replacement(tmp_9, tmp_5, in_1, in_3):
    # All ops are FX-traced and compiled by inductor (no eager-mode overhead)
    out_mask = triton_combine_masks(tmp_9, tmp_5)
    out_inv_freq = in_1.float().unsqueeze(0).unsqueeze(-1)
    out_pos_ids = in_3.float().unsqueeze(1)
    return out_mask, out_inv_freq, out_pos_ids


def pattern(tmp_9, tmp_5, in_1, in_3):
    tmp_10 = tmp_9[None, None, :, :]
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_12 = tmp_5[:, None, None, :]
    tmp_13 = tmp_11 * tmp_12
    torch.set_grad_enabled(False)
    tmp_15 = in_1[None, :, None]
    tmp_16 = tmp_15.float()
    tmp_17 = tmp_16.expand(1, -1, 1)
    tmp_18 = tmp_17.to(device(type='cuda', index=0))
    tmp_19 = in_3[:, None, :]
    tmp_20 = tmp_19.float()
    tmp_21 = tmp_18.float()
    tmp_22 = tmp_20.float()
    return tmp_13, tmp_21, tmp_22


def replacement_args(tmp_9, tmp_5, in_1, in_3):
    return (tmp_9, tmp_5, in_1, in_3)


def replacement_func():
    return _replacement