import torch
import triton
import triton.language as tl


@triton.jit
def _add_layernorm_kernel(
    a_ptr,
    b_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_size,
    stride_a0,
    stride_a1,
    stride_a2,
    stride_b0,
    stride_b1,
    stride_b2,
    stride_out0,
    stride_out1,
    stride_out2,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = batch_size * seq_len
    if pid >= rows:
        return

    batch = pid // seq_len
    s = pid % seq_len

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size

    a_ptrs = a_ptr + batch * stride_a0 + s * stride_a1 + cols * stride_a2
    b_ptrs = b_ptr + batch * stride_b0 + s * stride_b1 + cols * stride_b2

    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    x = tl.cast(a, tl.float32) + tl.cast(b, tl.float32)

    mean = tl.sum(x, axis=0) / hidden_size
    centered = x - mean
    var = tl.sum(centered * centered, axis=0) / hidden_size
    inv_std = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    y = centered * inv_std
    y = y * tl.cast(weight, tl.float32) + tl.cast(bias, tl.float32)

    out_ptrs = out_ptr + batch * stride_out0 + s * stride_out1 + cols * stride_out2
    tl.store(out_ptrs, y, mask=mask)


@triton.jit
def _embed_add_layernorm_kernel(
    input_ids_ptr,
    word_weight_ptr,
    pos_weight_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_size,
    input_stride_b,
    input_stride_s,
    word_stride_row,
    word_stride_col,
    pos_stride_row,
    pos_stride_col,
    out_stride_b,
    out_stride_s,
    out_stride_h,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = batch_size * seq_len
    if pid >= rows:
        return

    b = pid // seq_len
    s = pid % seq_len
    token_id = tl.load(input_ids_ptr + b * input_stride_b + s * input_stride_s)
    pos_id = s + 2

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size
    word_ptrs = word_weight_ptr + token_id * word_stride_row + cols * word_stride_col
    pos_ptrs = pos_weight_ptr + pos_id * pos_stride_row + cols * pos_stride_col

    word_vals = tl.load(word_ptrs, mask=mask, other=0.0)
    pos_vals = tl.load(pos_ptrs, mask=mask, other=0.0)
    x = tl.cast(word_vals, tl.float32) + tl.cast(pos_vals, tl.float32)

    mean = tl.sum(x, axis=0) / hidden_size
    centered = x - mean
    var = tl.sum(centered * centered, axis=0) / hidden_size
    inv_std = tl.rsqrt(var + eps)

    weight = tl.load(ln_weight_ptr + cols, mask=mask, other=1.0)
    bias = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0)
    y = centered * inv_std
    y = y * tl.cast(weight, tl.float32) + tl.cast(bias, tl.float32)

    out_ptrs = out_ptr + b * out_stride_b + s * out_stride_s + cols * out_stride_h
    tl.store(out_ptrs, y, mask=mask)



@triton.jit
def _mask_kernel(
    input_ids_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ids_ptr + offsets, mask=mask, other=0)
    y = tl.where(x == 1, -3.4028234663852886e+38, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _mask_kernel_strided(
    input_ids_ptr,
    out_ptr,
    batch_size,
    seq_len,
    in_stride0,
    in_stride1,
    out_stride0,
    out_stride1,
    out_stride2,
    out_stride3,
):
    pid = tl.program_id(0)
    total = batch_size * seq_len
    if pid >= total:
        return
    b = pid // seq_len
    s = pid % seq_len
    x = tl.load(input_ids_ptr + b * in_stride0 + s * in_stride1)
    y = tl.where(x == 1, -3.4028234663852886e+38, 0.0)
    tl.store(out_ptr + b * out_stride0 + s * out_stride3, y)


def _pick_block_size_and_warps(hidden_size):
    if hidden_size <= 32:
        return 32, 1
    if hidden_size <= 64:
        return 64, 1
    if hidden_size <= 128:
        return 128, 2
    if hidden_size <= 256:
        return 256, 4
    if hidden_size <= 512:
        return 512, 4
    return 1024, 8


@torch.fx.wrap
def ernie_dispatch(*args):
    route = args[-1]
    if route == "identity":
        x, _ = args
        return x



    if route == "addln":
        a, b, ln_bias, ln_weight, _ = args
        batch_size = a.shape[0]
        seq_len = a.shape[1]
        hidden_size = a.shape[2]
        out = torch.empty_like(a)
        block_size, num_warps = _pick_block_size_and_warps(hidden_size)
        _add_layernorm_kernel[(batch_size * seq_len,)](
            a,
            b,
            ln_weight,
            ln_bias,
            out,
            batch_size,
            seq_len,
            hidden_size,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            1e-5,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        return out

    if route == "embedln":
        input_ids, ln_bias, ln_weight, pos_weight, word_weight, _ = args
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        hidden_size = ln_weight.shape[0]
        out = torch.empty(
            (batch_size, seq_len, hidden_size),
            device=word_weight.device,
            dtype=word_weight.dtype,
        )
        block_size, num_warps = _pick_block_size_and_warps(hidden_size)
        _embed_add_layernorm_kernel[(batch_size * seq_len,)](
            input_ids,
            word_weight,
            pos_weight,
            ln_weight,
            ln_bias,
            out,
            batch_size,
            seq_len,
            hidden_size,
            input_ids.stride(0),
            input_ids.stride(1),
            word_weight.stride(0),
            word_weight.stride(1),
            pos_weight.stride(0),
            pos_weight.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            1e-5,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        return out

    if route == "mask":
        input_ids, _ = args
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        out = torch.empty(
            (batch_size, 1, 1, seq_len),
            device=input_ids.device,
            dtype=torch.float32,
        )
        _mask_kernel_strided[(batch_size * seq_len,)](
            input_ids,
            out,
            batch_size,
            seq_len,
            input_ids.stride(0),
            input_ids.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
        )
        return out

    input_ids, ln_bias, ln_weight, pos_weight, word_weight, _ = args
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    hidden_size = ln_weight.shape[0]
    out = torch.empty(
        (batch_size, seq_len, hidden_size),
        device=word_weight.device,
        dtype=word_weight.dtype,
    )
    mask = torch.empty(
        (batch_size, 1, 1, seq_len),
        device=input_ids.device,
        dtype=torch.float32,
    )
    block_size, num_warps = _pick_block_size_and_warps(hidden_size)
    grid = (batch_size * seq_len,)

    _ernie_embeddings_layernorm_and_mask_kernel[grid](
        input_ids,
        word_weight,
        pos_weight,
        ln_weight,
        ln_bias,
        out,
        mask,
        batch_size,
        seq_len,
        hidden_size,
        input_ids.stride(0),
        input_ids.stride(1),
        word_weight.stride(0),
        word_weight.stride(1),
        pos_weight.stride(0),
        pos_weight.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        1e-5,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out, mask


@triton.jit
def _ernie_embeddings_layernorm_and_mask_kernel(
    input_ids_ptr,
    word_weight_ptr,
    pos_weight_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    out_ptr,
    mask_ptr,
    batch_size,
    seq_len,
    hidden_size,
    input_stride_b,
    input_stride_s,
    word_stride_row,
    word_stride_col,
    pos_stride_row,
    pos_stride_col,
    out_stride_b,
    out_stride_s,
    out_stride_h,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = batch_size * seq_len
    if pid >= total_rows:
        return

    b = pid // seq_len
    s = pid % seq_len

    input_offset = b * input_stride_b + s * input_stride_s
    token_id = tl.load(input_ids_ptr + input_offset)

    mask_value = tl.where(token_id == 1, -3.4028234663852886e+38, 0.0)
    tl.store(mask_ptr + b * seq_len + s, mask_value)

    pos_id = s + 2
    cols = tl.arange(0, BLOCK_SIZE)
    col_mask = cols < hidden_size

    word_ptrs = word_weight_ptr + token_id * word_stride_row + cols * word_stride_col
    pos_ptrs = pos_weight_ptr + pos_id * pos_stride_row + cols * pos_stride_col

    word_vals = tl.load(word_ptrs, mask=col_mask, other=0.0)
    pos_vals = tl.load(pos_ptrs, mask=col_mask, other=0.0)

    x = tl.cast(word_vals, tl.float32) + tl.cast(pos_vals, tl.float32)
    mean = tl.sum(x, axis=0) / hidden_size
    centered = x - mean
    var = tl.sum(centered * centered, axis=0) / hidden_size
    inv_std = tl.rsqrt(var + eps)

    gamma = tl.load(ln_weight_ptr + cols, mask=col_mask, other=1.0)
    beta = tl.load(ln_bias_ptr + cols, mask=col_mask, other=0.0)
    y = centered * inv_std
    y = y * tl.cast(gamma, tl.float32) + tl.cast(beta, tl.float32)

    out_ptrs = out_ptr + b * out_stride_b + s * out_stride_s + cols * out_stride_h
    tl.store(out_ptrs, y, mask=col_mask)


def replacement_func():
    return ernie_dispatch