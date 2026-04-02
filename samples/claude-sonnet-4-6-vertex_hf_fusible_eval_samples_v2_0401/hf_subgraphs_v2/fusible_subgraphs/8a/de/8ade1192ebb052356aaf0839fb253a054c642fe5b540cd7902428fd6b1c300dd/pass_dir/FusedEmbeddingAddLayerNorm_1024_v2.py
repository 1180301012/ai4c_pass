import torch
from torch import device
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_9 = in_4.unsqueeze(0)
    tmp_10 = tmp_9 + 2
    tmp_11 = torch.nn.functional.embedding(tmp_10, in_1, None, None, 2.0, False, False)
    tmp_12 = tmp_11.to(device(type='cuda', index=0))
    tmp_13 = in_0 + tmp_12
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (1024,), in_3, in_2, 1e-05)
    return tmp_14


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 16}),
        triton.Config({'BLOCK_D': 32}),
        triton.Config({'BLOCK_D': 64}),
        triton.Config({'BLOCK_D': 128}),
        triton.Config({'BLOCK_D': 256}),
        triton.Config({'BLOCK_D': 512}),
        triton.Config({'BLOCK_D': 1024}),
    ],
    key=['D'],
)
@triton.jit
def embedding_add_layernorm_kernel_1024(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    in_3_ptr,
    in_4_ptr,
    out_ptr,
    D,
    eps: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    pos_idx = tl.load(in_4_ptr + pid)
    emb_idx = pos_idx + 2

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    x0_f32 = tl.load(in_0_ptr + pid * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    x1_f32 = tl.load(in_1_ptr + emb_idx * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    x = x0_f32 + x1_f32

    # Layer norm
    mean = tl.sum(tl.where(d_mask, x, 0.0), axis=0) / D
    diff = tl.where(d_mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = tl.rsqrt(var + eps)
    x_norm = diff * rstd

    weight = tl.load(in_3_ptr + d_offsets, mask=d_mask, other=1.0).to(tl.float32)
    bias   = tl.load(in_2_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    out_f32 = x_norm * weight + bias

    if IS_BF16:
        tl.store(out_ptr + pid * D + d_offsets, out_f32.to(tl.bfloat16), mask=d_mask)
    else:
        tl.store(out_ptr + pid * D + d_offsets, out_f32.to(tl.float16), mask=d_mask)


@torch.fx.wrap
def fused_embedding_add_layernorm_1024(in_0, in_1, in_2, in_3, in_4):
    B, N, D = in_0.shape
    BN = B * N
    out = torch.empty_like(in_0)
    IS_BF16 = (in_0.dtype == torch.bfloat16)
    grid = (BN,)
    embedding_add_layernorm_kernel_1024[grid](
        in_0.view(BN, D),
        in_1,
        in_2,
        in_3,
        in_4,
        out.view(BN, D),
        D=D,
        eps=1e-5,
        IS_BF16=IS_BF16,
    )
    return out


def replacement_func():
    return fused_embedding_add_layernorm_1024