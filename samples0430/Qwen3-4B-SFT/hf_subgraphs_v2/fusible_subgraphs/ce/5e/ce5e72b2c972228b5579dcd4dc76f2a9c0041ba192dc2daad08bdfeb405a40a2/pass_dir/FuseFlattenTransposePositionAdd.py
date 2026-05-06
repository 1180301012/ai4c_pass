import torch
import triton
import triton.language as tl
from torch import device


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512,  'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 2048, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 4096, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE': 512,  'num_warps': 8}),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE': 2048, 'num_warps': 8}),
    ],
    key=['N_total'],   # key only on N_total (int); constexpr HW allows per-CTA
)
@triton.jit
def flatten_transpose_add_kernel(
    conv_out_ptr,   # [1, C, H, W] — conv output, contiguous, row-major (c, hw)
    pos_ids_ptr,    # [1, S, C]    — position embeddings (on CPU)
    out_ptr,        # [1, S, C]    — output on CUDA
    N_total,        # S * C  (= 1568 * 768 = 1204224 for target)
    HW_STRIDE: tl.constexpr,   # H*W = 196 baked in at JIT compile
    BLOCK_SIZE: tl.constexpr,
):
    """
    1-D grid over N_total = S*C elements.
    Each element i maps to:
        c = i // HW_STRIDE   (channel,  0..C-1)
        x = i %  HW_STRIDE   (spatial, 0..HW-1)
    conv_out[c, x] lives at address  c*HW_STRIDE + x
    pos_ids[i]    lives at address  i  (stride-1, perfectly coalesced)
    """
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_total

    # constexpr HW_STRIDE → LLVM magic-number division in PTX
    c = offs // HW_STRIDE   # channel index
    x = offs %  HW_STRIDE   # spatial  index

    conv_vals = tl.load(conv_out_ptr + c * HW_STRIDE + x, mask=mask, other=0.0)
    pos_vals  = tl.load(pos_ids_ptr  + offs,               mask=mask, other=0.0)

    result = pos_vals.to(conv_vals.dtype) + conv_vals
    tl.store(out_ptr + offs, result, mask=mask)


@torch.fx.wrap
def triton_flatten_transpose_add(conv_out, pos_ids):
    """
    Fused kernel for:
        flatten(2) + transpose(1,2) + detach + type_as + to(cuda) + add

    conv_out : [B, C, H, W] CUDA tensor (output of conv3d)
    pos_ids  : [B, S, C]    CPU tensor  (position embeddings, on CPU)
    """
    C    = conv_out.shape[1]
    HW   = conv_out.shape[2] * conv_out.shape[3]   # 196 for target models
    B    = 1
    N    = pos_ids.numel()                          # S*C for batch=1

    out  = torch.empty(B, N // (B * C), C, dtype=conv_out.dtype, device='cuda')

    # 1-D grid: autotuner selects best BLOCK_SIZE per (N_total, HW_STRIDE)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    flatten_transpose_add_kernel[grid](
        conv_out, pos_ids, out,
        N,
        HW_STRIDE=HW,   # constexpr — JIT-bakes HW_STRIDE=196
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / argument extraction
# ---------------------------------------------------------------------------

def pattern(conv_out, pos_ids):
    """
    Matches the subgraph:
        tmp_4 = conv_out.flatten(2)
        tmp_5 = tmp_4.transpose(1, 2)
        tmp_6 = pos_ids.detach()
        tmp_7 = tmp_6.type_as(tmp_5)
        tmp_8 = tmp_7.to(device=device(type='cuda', index=0), copy=True)
        tmp_9 = tmp_5 + tmp_8
        return (tmp_9,)
    """
    tmp_4 = conv_out.flatten(2)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = pos_ids.detach()
    tmp_7 = tmp_6.type_as(tmp_5)
    tmp_8 = tmp_7.to(device=device(type='cuda', index=0), copy=True)
    tmp_9 = tmp_5 + tmp_8
    return (tmp_9,)


def replacement_args(conv_out, pos_ids):
    return (conv_out, pos_ids)


def replacement_func():
    return triton_flatten_transpose_add