import torch
import triton
import triton.language as tl
from torch import device


def pattern(conv_out, pos_embed):
    tmp_4 = conv_out.flatten(2)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = pos_embed.detach()
    tmp_7 = tmp_6.type_as(tmp_5)
    tmp_8 = tmp_7.to(device=device(type='cuda', index=0), copy=True)
    tmp_9 = tmp_5 + tmp_8
    return tmp_9


def replacement_args(conv_out, pos_embed):
    return (conv_out, pos_embed)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_N': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_C': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    ],
    key=['C', 'N'],
)
@triton.jit
def fused_transpose_add_kernel(
    conv_out_ptr,
    pos_embed_ptr,
    out_ptr,
    C,
    N,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    n_mask = n_offs < N
    c_mask = c_offs < C
    mask = n_mask[:, None] & c_mask[None, :]  # [BLOCK_N, BLOCK_C]

    # conv_out: [C, N] layout, element (c, n) at c * N + n
    conv_idx = c_offs[None, :] * N + n_offs[:, None]  # [BLOCK_N, BLOCK_C]
    conv_vals = tl.load(conv_out_ptr + conv_idx, mask=mask, other=0.0)

    # pos_embed: [N, C] layout, element (n, c) at n * C + c
    pos_idx = n_offs[:, None] * C + c_offs[None, :]  # [BLOCK_N, BLOCK_C]
    pos_vals = tl.load(pos_embed_ptr + pos_idx, mask=mask, other=0.0)

    # Fused add
    result = conv_vals + pos_vals

    # Store output: [N, C] layout
    tl.store(out_ptr + pos_idx, result, mask=mask)


@torch.fx.wrap
def fused_transpose_add(conv_out, pos_embed):
    B = conv_out.shape[0]
    C = conv_out.shape[1]
    N = 1
    for i in range(2, len(conv_out.shape)):
        N *= conv_out.shape[i]

    # Move pos_embed to GPU with correct dtype
    pos_embed_gpu = torch.as_tensor(pos_embed, device=conv_out.device, dtype=conv_out.dtype)

    # Allocate output [B, N, C]
    out = torch.empty(B, N, C, device=conv_out.device, dtype=conv_out.dtype)

    # Grid dimensions
    grid = lambda meta: (
        (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],
        (C + meta['BLOCK_C'] - 1) // meta['BLOCK_C'],
    )

    fused_transpose_add_kernel[grid](
        conv_out,
        pos_embed_gpu,
        out,
        C, N,
    )

    return out


def replacement_func():
    return fused_transpose_add