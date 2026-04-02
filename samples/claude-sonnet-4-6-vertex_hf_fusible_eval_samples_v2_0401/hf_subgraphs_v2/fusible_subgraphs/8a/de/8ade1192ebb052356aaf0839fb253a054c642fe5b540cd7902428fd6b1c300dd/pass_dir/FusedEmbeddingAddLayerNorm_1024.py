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


@triton.jit
def embedding_add_layernorm_kernel_1024(
    in_0_ptr,   # [B*N, D] flattened input embeddings
    in_1_ptr,   # [V, D] position embedding weight table
    in_2_ptr,   # [D] layer norm bias
    in_3_ptr,   # [D] layer norm weight
    in_4_ptr,   # [N] cache positions (int64)
    out_ptr,    # [B*N, D] flattened output
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    eps: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid = tl.program_id(0)  # token index in [0, B*N)

    # Load position index and add 2 for embedding offset
    pos_idx = tl.load(in_4_ptr + pid)
    emb_idx = pos_idx + 2

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    # Load input embedding and position embedding, cast to float32
    x0_f32 = tl.load(in_0_ptr + pid * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    x1_f32 = tl.load(in_1_ptr + emb_idx * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

    # Add
    x = x0_f32 + x1_f32

    # Layer norm: mean over D valid elements
    mean = tl.sum(tl.where(d_mask, x, 0.0), axis=0) / D
    # Centered difference (zero out invalid lanes)
    diff = tl.where(d_mask, x - mean, 0.0)
    # Variance
    var = tl.sum(diff * diff, axis=0) / D
    rstd = tl.rsqrt(var + eps)
    x_norm = diff * rstd

    # Affine transform
    weight = tl.load(in_3_ptr + d_offsets, mask=d_mask, other=1.0).to(tl.float32)
    bias   = tl.load(in_2_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    out_f32 = x_norm * weight + bias

    # Store back in original dtype
    if IS_BF16:
        tl.store(out_ptr + pid * D + d_offsets, out_f32.to(tl.bfloat16), mask=d_mask)
    else:
        tl.store(out_ptr + pid * D + d_offsets, out_f32.to(tl.float16), mask=d_mask)


@torch.fx.wrap
def fused_embedding_add_layernorm_1024(in_0, in_1, in_2, in_3, in_4):
    D = 1024
    BLOCK_D = 1024
    B, N, _ = in_0.shape
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
        BLOCK_D=BLOCK_D,
        eps=1e-5,
        IS_BF16=IS_BF16,
    )
    return out


def replacement_func():
    return fused_embedding_add_layernorm_1024