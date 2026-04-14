import torch
from torch import device
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_J': 64}, num_warps=2),
        triton.Config({'BLOCK_J': 128}, num_warps=4),
        triton.Config({'BLOCK_J': 256}, num_warps=4),
        triton.Config({'BLOCK_J': 512}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def causal_mask_kernel_N512(
    in0_ptr,
    in2_ptr,
    out_ptr,
    B,
    N,
    BLOCK_J: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.program_id(1)
    j_tile = tl.program_id(2)

    j_offsets = j_tile * BLOCK_J + tl.arange(0, BLOCK_J)
    mask_j = j_offsets < N

    # Load cache position in_2[i]
    in2_val = tl.load(in2_ptr + i).to(tl.int64)

    # Load attention mask in_0[b, j]
    in0 = tl.load(in0_ptr + b * N + j_offsets, mask=mask_j, other=0).to(tl.int64)

    # Causal mask: j <= in_2[i]
    causal = j_offsets.to(tl.int64) <= in2_val

    # Attention mask: in_0[b, j] != 0
    attn = in0 != 0

    # Combined mask stored as int8 (bool)
    out_val = (causal & attn).to(tl.int8)

    # Store to out[b, 0, i, j]
    tl.store(out_ptr + b * N * N + i * N + j_offsets, out_val, mask=mask_j)


@torch.fx.wrap
def causal_mask_wrapper_N512(in_0, in_2):
    B = in_0.shape[0]
    N = 512
    out = torch.empty((B, 1, N, N), dtype=torch.bool, device=in_0.device)
    grid = lambda meta: (B, N, triton.cdiv(N, meta['BLOCK_J']))
    causal_mask_kernel_N512[grid](in_0, in_2, out, B, N)
    return out


def pattern(in_0, in_2):
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    tmp_3 = torch.arange(512, device=device(type='cuda', index=0))
    tmp_3 = tmp_3 + 0
    tmp_4 = tmp_3
    tmp_5 = tmp_2[(slice(None, None, None), tmp_4)]
    tmp_6 = torch.arange(512, device=device(type='cuda', index=0))
    tmp_6 = tmp_6 + 0
    tmp_7 = tmp_6
    tmp_8 = in_2.view(-1, 1)
    tmp_9 = tmp_7 <= tmp_8
    tmp_10 = tmp_9[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_12 = tmp_5[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_11 * tmp_12
    return tmp_13


def replacement_args(in_0, in_2):
    return (in_0, in_2)


def replacement_func():
    return causal_mask_wrapper_N512