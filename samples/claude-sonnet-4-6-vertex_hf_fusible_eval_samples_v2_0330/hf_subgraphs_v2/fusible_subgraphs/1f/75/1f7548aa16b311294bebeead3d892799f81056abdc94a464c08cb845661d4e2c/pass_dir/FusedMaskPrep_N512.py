import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def fused_mask_kernel_n512(
    in0_ptr,   # [B, N] int64 attention mask
    in2_ptr,   # [N] int64 cache_position
    out_ptr,   # [B, N, N] bool (viewed as [B, 1, N, N] after)
    N,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.program_id(1)

    # Load cache_position[i]
    cache_i = tl.load(in2_ptr + i)

    j_offs = tl.arange(0, BLOCK_N)
    mask = j_offs < N

    # Load in0[b, j] and convert to bool
    in0_val = tl.load(in0_ptr + b * N + j_offs, mask=mask, other=0)
    in0_bool = in0_val != 0

    # Causal mask: j <= cache_position[i]
    causal = j_offs <= cache_i

    # Combined mask
    out_val = in0_bool & causal

    tl.store(out_ptr + b * N * N + i * N + j_offs, out_val, mask=mask)


@torch.fx.wrap
def fused_mask_invfreq_posids_n512(in_0, in_1, in_2, in_3):
    B, N = in_0.shape

    # Compute fused mask: [B, N, N] -> view to [B, 1, N, N]
    out_mask = torch.empty(B, N, N, dtype=torch.bool, device=in_0.device)

    grid = (B, N)
    fused_mask_kernel_n512[grid](
        in_0, in_2, out_mask,
        N,
        BLOCK_N=512,
    )

    out_mask = out_mask.view(B, 1, N, N)

    # inv_freq: [D] -> [1, D, 1] float32
    out_inv_freq = in_1.float().unsqueeze(0).unsqueeze(-1)

    # position_ids: [B_pos, S] -> [B_pos, 1, S] float32
    out_pos_ids = in_3.float().unsqueeze(1)

    return out_mask, out_inv_freq, out_pos_ids


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    tmp_3 = torch.arange(512, device=device(type='cuda', index=0))
    tmp_3 += 0
    tmp_5 = tmp_2[:, tmp_3]
    tmp_6 = torch.arange(512, device=device(type='cuda', index=0))
    tmp_6 += 0
    tmp_8 = in_2.view(-1, 1)
    tmp_9 = torch.ops.aten.le.Tensor(tmp_6, tmp_8)
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


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_mask_invfreq_posids_n512