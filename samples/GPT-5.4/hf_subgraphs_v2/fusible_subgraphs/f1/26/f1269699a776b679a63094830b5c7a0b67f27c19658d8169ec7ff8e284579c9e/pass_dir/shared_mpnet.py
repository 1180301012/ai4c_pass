import torch
import triton
import triton.language as tl

from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


@triton.jit
def fused_embed_add_layernorm_kernel(
    input_ids_ptr,
    position_ids_ptr,
    word_weight_ptr,
    pos_weight_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    input_stride_0,
    input_stride_1,
    position_id_stride_0,
    position_id_stride_1,
    word_stride_0,
    word_stride_1,
    pos_stride_0,
    pos_stride_1,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    seq_len,
    hidden,
    eps,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // seq_len
    s = pid % seq_len

    word_id = tl.load(input_ids_ptr + b * input_stride_0 + s * input_stride_1)
    pos_id = tl.load(position_ids_ptr + b * position_id_stride_0 + s * position_id_stride_1)

    offs = tl.arange(0, BLOCK_H)
    mask = offs < hidden

    word_vals = tl.load(
        word_weight_ptr + word_id * word_stride_0 + offs * word_stride_1,
        mask=mask,
        other=0.0,
    )
    pos_vals = tl.load(
        pos_weight_ptr + pos_id * pos_stride_0 + offs * pos_stride_1,
        mask=mask,
        other=0.0,
    )

    x = (word_vals + pos_vals).to(tl.float32)
    mean = tl.sum(x, axis=0) / hidden
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / hidden
    inv_std = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = centered * inv_std * gamma + beta

    tl.store(
        out_ptr + b * out_stride_0 + s * out_stride_1 + offs * out_stride_2,
        y,
        mask=mask,
    )


@torch.fx.wrap
def fused_mpnet_dispatch(in_0, in_1, in_2, in_3, in_4, in_5):
    input_ids = unwrap_tensor(in_0)
    ln_bias = unwrap_tensor(in_1)
    ln_weight = unwrap_tensor(in_2)
    pos_weight = unwrap_tensor(in_3)
    word_weight = unwrap_tensor(in_4)
    position_ids = unwrap_tensor(in_5)

    batch = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    hidden = ln_weight.shape[0]
    eps = 1e-12 if hidden == 64 else 1e-5

    out = torch.empty(
        (batch, seq_len, hidden),
        device=word_weight.device,
        dtype=word_weight.dtype,
    )

    if hidden <= 64:
        block_h = 64
        num_warps = 2
    else:
        block_h = 1024
        num_warps = 8

    grid = (batch * seq_len,)
    fused_embed_add_layernorm_kernel[grid](
        input_ids,
        position_ids,
        word_weight,
        pos_weight,
        ln_weight,
        ln_bias,
        out,
        input_ids.stride(0),
        input_ids.stride(1),
        position_ids.stride(0),
        position_ids.stride(1),
        word_weight.stride(0),
        word_weight.stride(1),
        pos_weight.stride(0),
        pos_weight.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        seq_len,
        hidden,
        eps,
        BLOCK_H=block_h,
        num_warps=num_warps,
        num_stages=2,
    )

    return out