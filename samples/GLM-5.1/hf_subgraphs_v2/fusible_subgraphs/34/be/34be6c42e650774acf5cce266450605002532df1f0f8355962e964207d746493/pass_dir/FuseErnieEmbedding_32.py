import torch
import triton
import triton.language as tl


def pattern(word_ids, word_emb, pos_ids, pos_emb, ln_weight, ln_bias):
    # Embedding computation
    tmp_10 = torch.nn.functional.embedding(word_ids, word_emb, 1, None, 2.0, False, False)
    tmp_15 = torch.nn.functional.embedding(pos_ids, pos_emb, 1, None, 2.0, False, False)
    # Combine and normalize
    tmp_16 = tmp_10 + tmp_15
    tmp_17 = torch.nn.functional.layer_norm(tmp_16, (32,), ln_weight, ln_bias, 1e-05)
    tmp_18 = torch.nn.functional.dropout(tmp_17, 0.1, False, False)
    return tmp_18


def replacement_args(word_ids, word_emb, pos_ids, pos_emb, ln_weight, ln_bias):
    return (word_ids, word_emb, pos_ids, pos_emb, ln_weight, ln_bias, "route_32")


@triton.jit
def fused_ernie_emb_kernel_768(
    input_ids_ptr,
    word_emb_ptr,
    pos_emb_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    output_ptr,
    seq_len,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    seq_idx = pid  # batch_size=1

    # Load input_id
    input_id = tl.load(input_ids_ptr + seq_idx)

    # Compute position index: seq_idx + 2
    pos_idx = seq_idx + 2

    # Load word embedding row
    offsets = tl.arange(0, BLOCK_SIZE)
    h_mask = offsets < H
    word_row = tl.load(word_emb_ptr + input_id * H + offsets, mask=h_mask, other=0.0)

    # Load position embedding row
    pos_row = tl.load(pos_emb_ptr + pos_idx * H + offsets, mask=h_mask, other=0.0)

    # Add embeddings (compute in float32)
    combined = (word_row + pos_row).to(tl.float32)

    # Layer norm: compute mean
    mean = tl.sum(combined, axis=0) / H

    # Compute variance
    diff = combined - mean
    var = tl.sum(diff * diff, axis=0) / H

    # Normalize
    rstd = 1.0 / tl.sqrt(var + 1e-05)
    normalized = diff * rstd

    # Load weight and bias
    ln_w = tl.load(ln_weight_ptr + offsets, mask=h_mask, other=1.0).to(tl.float32)
    ln_b = tl.load(ln_bias_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)

    # Apply weight and bias
    result = normalized * ln_w + ln_b

    # Store result
    tl.store(output_ptr + pid * H + offsets, result, mask=h_mask)


@triton.jit
def fused_ernie_emb_kernel_32(
    input_ids_ptr,
    word_emb_ptr,
    pos_emb_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    output_ptr,
    seq_len,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    seq_idx = pid  # batch_size=1

    # Load input_id
    input_id = tl.load(input_ids_ptr + seq_idx)

    # Compute position index: seq_idx + 2
    pos_idx = seq_idx + 2

    # Load word embedding row
    offsets = tl.arange(0, BLOCK_SIZE)
    h_mask = offsets < H
    word_row = tl.load(word_emb_ptr + input_id * H + offsets, mask=h_mask, other=0.0)

    # Load position embedding row
    pos_row = tl.load(pos_emb_ptr + pos_idx * H + offsets, mask=h_mask, other=0.0)

    # Add embeddings (compute in float32)
    combined = (word_row + pos_row).to(tl.float32)

    # Layer norm: compute mean
    mean = tl.sum(combined, axis=0) / H

    # Compute variance
    diff = combined - mean
    var = tl.sum(diff * diff, axis=0) / H

    # Normalize
    rstd = 1.0 / tl.sqrt(var + 1e-05)
    normalized = diff * rstd

    # Load weight and bias
    ln_w = tl.load(ln_weight_ptr + offsets, mask=h_mask, other=1.0).to(tl.float32)
    ln_b = tl.load(ln_bias_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)

    # Apply weight and bias
    result = normalized * ln_w + ln_b

    # Store result
    tl.store(output_ptr + pid * H + offsets, result, mask=h_mask)


def _fused_768(word_ids, word_emb, pos_ids, pos_emb, ln_weight, ln_bias):
    batch_size, seq_len = word_ids.shape
    dtype = word_emb.dtype
    dev = word_emb.device

    # Create output tensor
    output = torch.empty((batch_size, seq_len, 768), dtype=dtype, device=dev)

    # Launch kernel
    n_positions = batch_size * seq_len
    grid = (n_positions,)

    fused_ernie_emb_kernel_768[grid](
        input_ids_ptr=word_ids,
        word_emb_ptr=word_emb,
        pos_emb_ptr=pos_emb,
        ln_weight_ptr=ln_weight,
        ln_bias_ptr=ln_bias,
        output_ptr=output,
        seq_len=seq_len,
        H=768,
        BLOCK_SIZE=1024,
    )

    return output


def _fused_32(word_ids, word_emb, pos_ids, pos_emb, ln_weight, ln_bias):
    batch_size, seq_len = word_ids.shape
    dtype = word_emb.dtype
    dev = word_emb.device

    # Create output tensor
    output = torch.empty((batch_size, seq_len, 32), dtype=dtype, device=dev)

    # Launch kernel
    n_positions = batch_size * seq_len
    grid = (n_positions,)

    fused_ernie_emb_kernel_32[grid](
        input_ids_ptr=word_ids,
        word_emb_ptr=word_emb,
        pos_emb_ptr=pos_emb,
        ln_weight_ptr=ln_weight,
        ln_bias_ptr=ln_bias,
        output_ptr=output,
        seq_len=seq_len,
        H=32,
        BLOCK_SIZE=64,
    )

    return output


@torch.fx.wrap
def fused_ernie_emb_dispatch(word_ids, word_emb, pos_ids, pos_emb, ln_weight, ln_bias, route):
    if route == "route_768":
        return _fused_768(word_ids, word_emb, pos_ids, pos_emb, ln_weight, ln_bias)
    elif route == "route_32":
        return _fused_32(word_ids, word_emb, pos_ids, pos_emb, ln_weight, ln_bias)
    else:
        raise ValueError(f"Unknown route: {route}")


def replacement_func():
    return fused_ernie_emb_dispatch