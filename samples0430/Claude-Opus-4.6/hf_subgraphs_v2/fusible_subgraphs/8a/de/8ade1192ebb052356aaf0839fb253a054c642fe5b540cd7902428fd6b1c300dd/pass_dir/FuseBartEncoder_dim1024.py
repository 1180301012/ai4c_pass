import torch
import triton
import triton.language as tl
from torch import device


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = in_5.to(torch.float32)
    tmp_5 = torch.tensor(1.0, dtype=torch.float32)
    tmp_6 = tmp_5 - tmp_4
    tmp_7 = tmp_6.to(torch.bool)
    tmp_8 = tmp_6.masked_fill(tmp_7, -3.4028234663852886e+38)
    tmp_9 = in_4.unsqueeze(0)
    tmp_10 = tmp_9 + 2
    tmp_11 = torch.nn.functional.embedding(tmp_10, in_1, None, None, 2.0, False, False)
    tmp_12 = tmp_11.to(device(type='cuda', index=0))
    tmp_13 = in_0 + tmp_12
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (1024,), in_3, in_2, 1e-05)
    tmp_15 = torch.nn.functional.dropout(tmp_14, p=0.1, training=False)
    return (tmp_8, tmp_15)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.jit
def mask_kernel(
    in_5_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    val = tl.load(in_5_ptr + offsets, mask=mask, other=0)
    # in_5 is int64 with values 0 or 1
    # Result: 0.0 if val == 1, -3.4028234663852886e+38 if val == 0
    val_f32 = val.to(tl.float32)
    result = 1.0 - val_f32
    # Where result != 0 (i.e., val was 0), fill with -3.4028234663852886e+38
    is_nonzero = result != 0.0
    result = tl.where(is_nonzero, -3.4028234663852886e+38, result)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def fused_embed_add_layernorm_kernel_1024(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    in_3_ptr,
    in_4_ptr,
    out_ptr,
    seq_len,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_idx = tl.program_id(0)

    # Load index for this row
    idx = tl.load(in_4_ptr + row_idx) + 2

    cols = tl.arange(0, BLOCK_D)
    mask_d = cols < D

    # Load in_0 row [D]
    x = tl.load(in_0_ptr + row_idx * D + cols, mask=mask_d, other=0.0)

    # Load embedding row [D]
    emb = tl.load(in_1_ptr + idx * D + cols, mask=mask_d, other=0.0)

    # Add embedding to input
    x = x + emb

    # Layer norm in float32
    x_f32 = x.to(tl.float32)
    mean = tl.sum(x_f32, axis=0) / D
    diff = x_f32 - mean
    var = tl.sum(diff * diff, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + 1e-05)
    normalized = diff * inv_std

    # Load weight and bias
    weight = tl.load(in_3_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)
    bias = tl.load(in_2_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)

    # Apply affine transform
    out = normalized * weight + bias

    # Store in original dtype
    tl.store(out_ptr + row_idx * D + cols, out.to(x.dtype), mask=mask_d)


@torch.fx.wrap
def fused_bart_encoder_1024(in_0, in_1, in_2, in_3, in_4, in_5):
    # Mask computation
    mask_numel = in_5.numel()
    MASK_BLOCK = 1024
    mask_out = torch.empty(in_5.shape, dtype=torch.float32, device=in_5.device)
    grid_mask = ((mask_numel + MASK_BLOCK - 1) // MASK_BLOCK,)
    mask_kernel[grid_mask](in_5, mask_out, mask_numel, BLOCK_SIZE=MASK_BLOCK)

    # Fused embedding + add + layer_norm
    batch_seq = in_0.shape[0] * in_0.shape[1]
    D = 1024
    BLOCK_D = 1024
    out = torch.empty_like(in_0)
    grid_ln = (batch_seq,)
    fused_embed_add_layernorm_kernel_1024[grid_ln](
        in_0, in_1, in_2, in_3, in_4, out,
        in_0.shape[1],
        D=D,
        BLOCK_D=BLOCK_D,
    )

    return (mask_out, out)


def replacement_func():
    return fused_bart_encoder_1024