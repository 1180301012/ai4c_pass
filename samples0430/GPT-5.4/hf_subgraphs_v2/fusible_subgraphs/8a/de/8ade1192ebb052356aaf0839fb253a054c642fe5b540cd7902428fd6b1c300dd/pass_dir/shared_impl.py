import torch
import triton
import triton.language as tl


NEG_INF = -3.4028234663852886e+38
LN_EPS = 1e-5


@triton.jit
def causal_mask_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=1)
    out = tl.where(x == 1, 0.0, NEG_INF)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def fused_embedding_add_layernorm_kernel(
    input_ptr,
    embed_ptr,
    gamma_ptr,
    beta_ptr,
    pos_ptr,
    out_ptr,
    input_row_stride,
    input_col_stride,
    embed_row_stride,
    embed_col_stride,
    gamma_stride,
    beta_stride,
    pos_stride,
    out_row_stride,
    out_col_stride,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    col_mask = cols < hidden_size

    pos = tl.load(pos_ptr + row * pos_stride) + 2

    input_row_ptr = input_ptr + row * input_row_stride + cols * input_col_stride
    embed_row_ptr = embed_ptr + pos * embed_row_stride + cols * embed_col_stride

    x = tl.load(input_row_ptr, mask=col_mask, other=0.0).to(tl.float32)
    e = tl.load(embed_row_ptr, mask=col_mask, other=0.0).to(tl.float32)
    v = x + e

    mean = tl.sum(v, axis=0) / hidden_size
    centered = v - mean
    var = tl.sum(centered * centered, axis=0) / hidden_size
    inv_std = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + cols * gamma_stride, mask=col_mask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + cols * beta_stride, mask=col_mask, other=0.0).to(tl.float32)
    y = centered * inv_std * gamma + beta

    out_row_ptr = out_ptr + row * out_row_stride + cols * out_col_stride
    tl.store(out_row_ptr, y, mask=col_mask)


@torch.fx.wrap
def fused_mask_embedding_add_layernorm(in_0, in_1, in_2, in_3, in_4, in_5):
    seq_len = in_4.numel()
    hidden_size = in_0.shape[-1]

    tmp_8 = torch.empty_like(in_5, dtype=torch.float32)
    tmp_15 = torch.empty_like(in_0)

    n_mask = in_5.numel()
    mask_block = 256
    mask_grid = ((n_mask + mask_block - 1) // mask_block,)
    causal_mask_kernel[mask_grid](
        in_5,
        tmp_8,
        n_mask,
        BLOCK_SIZE=mask_block,
    )

    if hidden_size <= 16:
        block_size = 16
        num_warps = 1
    elif hidden_size <= 128:
        block_size = 128
        num_warps = 2
    elif hidden_size <= 256:
        block_size = 256
        num_warps = 4
    elif hidden_size <= 512:
        block_size = 512
        num_warps = 4
    else:
        block_size = 1024
        num_warps = 8

    row_stride_in = in_0.stride(1)
    col_stride_in = in_0.stride(2)
    row_stride_out = tmp_15.stride(1)
    col_stride_out = tmp_15.stride(2)

    fused_embedding_add_layernorm_kernel[(seq_len,)](
        in_0,
        in_1,
        in_3,
        in_2,
        in_4,
        tmp_15,
        row_stride_in,
        col_stride_in,
        in_1.stride(0),
        in_1.stride(1),
        in_3.stride(0),
        in_2.stride(0),
        in_4.stride(0),
        row_stride_out,
        col_stride_out,
        hidden_size,
        LN_EPS,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=1,
    )

    return tmp_8, tmp_15


def replacement_func():
    return fused_mask_embedding_add_layernorm