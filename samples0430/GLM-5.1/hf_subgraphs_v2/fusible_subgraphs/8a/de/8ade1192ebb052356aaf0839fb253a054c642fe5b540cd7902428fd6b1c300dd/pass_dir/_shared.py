import torch
import triton
import triton.language as tl


# ============================================================
# Mask computation Triton kernel
# ============================================================
@triton.jit
def mask_compute_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    NEG_INF_VAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load int64 values
    in_val = tl.load(in_ptr + offsets, mask=mask)
    # Cast to float32
    in_val_f32 = in_val.to(tl.float32)
    # Compute 1.0 - in_val_f32 (same as tmp_5 - tmp_4)
    diff = 1.0 - in_val_f32
    # Convert to bool: True where diff != 0 (same as tmp_6.to(bool))
    is_nonzero = diff != 0.0
    # masked_fill: where is_nonzero, fill with NEG_INF_VAL; where not, keep diff
    out_val = tl.where(is_nonzero, NEG_INF_VAL, diff)
    tl.store(out_ptr + offsets, out_val, mask=mask)


@torch.fx.wrap
def _fused_mask_compute(in_5):
    n_elements = in_5.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty(in_5.shape, dtype=torch.float32, device=in_5.device)

    mask_compute_kernel[(num_programs,)](
        in_ptr=in_5,
        out_ptr=out,
        n_elements=n_elements,
        NEG_INF_VAL=-3.4028234663852886e+38,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


# ============================================================
# Embed + Add + LayerNorm Triton kernel
# ============================================================
@triton.jit
def fuse_embed_add_layernorm_kernel(
    inputs_ptr, stride_inputs_batch, stride_inputs_seq, stride_inputs_hidden,
    embed_w_ptr, stride_embed_pos, stride_embed_hidden,
    ln_weight_ptr, stride_ln_hidden,
    ln_bias_ptr, stride_ln_bias_hidden,
    cache_pos_ptr, stride_cache_pos,
    output_ptr, stride_output_batch, stride_output_seq, stride_output_hidden,
    seq_len,
    hidden_dim,
    embed_offset: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    if seq_idx >= seq_len:
        return

    # Load position index from cache_position and add offset
    pos_idx = tl.load(cache_pos_ptr + seq_idx * stride_cache_pos) + embed_offset

    # Compute row base pointers for this sequence position
    inputs_row_ptr = inputs_ptr + seq_idx * stride_inputs_seq
    embed_row_ptr = embed_w_ptr + pos_idx * stride_embed_pos
    output_row_ptr = output_ptr + seq_idx * stride_output_seq

    # First pass: compute mean of (input + embed) across hidden_dim
    mean_acc = 0.0
    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        valid_mask = offsets < hidden_dim
        input_val = tl.load(inputs_row_ptr + offsets * stride_inputs_hidden, mask=valid_mask, other=0.0).to(tl.float32)
        embed_val = tl.load(embed_row_ptr + offsets * stride_embed_hidden, mask=valid_mask, other=0.0).to(tl.float32)
        sum_val = input_val + embed_val
        mean_acc += tl.sum(sum_val * valid_mask.to(tl.float32))
    mean = mean_acc / hidden_dim

    # Second pass: compute variance
    var_acc = 0.0
    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        valid_mask = offsets < hidden_dim
        input_val = tl.load(inputs_row_ptr + offsets * stride_inputs_hidden, mask=valid_mask, other=0.0).to(tl.float32)
        embed_val = tl.load(embed_row_ptr + offsets * stride_embed_hidden, mask=valid_mask, other=0.0).to(tl.float32)
        sum_val = input_val + embed_val
        diff = sum_val - mean
        var_acc += tl.sum(diff * diff * valid_mask.to(tl.float32))
    var = var_acc / hidden_dim
    rstd = 1.0 / tl.sqrt(var + eps)

    # Third pass: normalize, apply weight/bias, and store
    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        valid_mask = offsets < hidden_dim
        input_val = tl.load(inputs_row_ptr + offsets * stride_inputs_hidden, mask=valid_mask, other=0.0).to(tl.float32)
        embed_val = tl.load(embed_row_ptr + offsets * stride_embed_hidden, mask=valid_mask, other=0.0).to(tl.float32)
        sum_val = input_val + embed_val
        w_val = tl.load(ln_weight_ptr + offsets * stride_ln_hidden, mask=valid_mask, other=1.0).to(tl.float32)
        b_val = tl.load(ln_bias_ptr + offsets * stride_ln_bias_hidden, mask=valid_mask, other=0.0).to(tl.float32)
        normalized = (sum_val - mean) * rstd
        output_val = normalized * w_val + b_val
        tl.store(output_row_ptr + offsets * stride_output_hidden, output_val, mask=valid_mask)


@torch.fx.wrap
def _fused_embed_add_layernorm(in_0, in_1, in_2, in_3, in_4):
    seq_len = in_4.shape[0]
    hidden_dim = in_0.shape[-1]

    output = torch.empty_like(in_0)

    BLOCK_SIZE = max(32, triton.next_power_of_2(hidden_dim))
    # Compute num_warps based on BLOCK_SIZE
    num_warps = min(8, max(1, BLOCK_SIZE // 32))

    grid = (seq_len,)

    fuse_embed_add_layernorm_kernel[grid](
        inputs_ptr=in_0,
        stride_inputs_batch=in_0.stride(0),
        stride_inputs_seq=in_0.stride(1),
        stride_inputs_hidden=in_0.stride(2),
        embed_w_ptr=in_1,
        stride_embed_pos=in_1.stride(0),
        stride_embed_hidden=in_1.stride(1),
        ln_weight_ptr=in_3,
        stride_ln_hidden=in_3.stride(0),
        ln_bias_ptr=in_2,
        stride_ln_bias_hidden=in_2.stride(0),
        cache_pos_ptr=in_4,
        stride_cache_pos=in_4.stride(0),
        output_ptr=output,
        stride_output_batch=output.stride(0),
        stride_output_seq=output.stride(1),
        stride_output_hidden=output.stride(2),
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        embed_offset=2,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return output


# ============================================================
# Dispatch wrapper - shared by all passes
# ============================================================
@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]

    if route == "mask":
        in_5 = args[0]
        return _fused_mask_compute(in_5)
    elif route == "embed_layernorm":
        in_0, in_1, in_2, in_3, in_4 = args[0], args[1], args[2], args[3], args[4]
        return _fused_embed_add_layernorm(in_0, in_1, in_2, in_3, in_4)
    else:
        raise ValueError(f"Unknown route: {route}")