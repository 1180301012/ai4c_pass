import torch
import triton
import triton.language as tl


# ========== Triton Kernel ==========

@triton.jit
def fused_emb_sum_ln_kernel(
    input_ids_ptr, token_type_ids_ptr, position_ids_ptr,
    word_emb_ptr, token_type_emb_ptr, position_emb_ptr,
    ln_weight_ptr, ln_bias_ptr,
    output_sum_ptr, output_ln_ptr,
    seq_len, hidden_size,
    input_ids_stride0, position_ids_stride0,
    eps_val,
    RETURN_BOTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len

    # Load index values (int64 -> int32)
    word_id = tl.load(input_ids_ptr + batch_idx * input_ids_stride0 + seq_idx).to(tl.int32)
    token_type_id = tl.load(token_type_ids_ptr + batch_idx * input_ids_stride0 + seq_idx).to(tl.int32)
    # position_ids has batch dim = 1, always use row 0
    position_id = tl.load(position_ids_ptr + seq_idx).to(tl.int32)

    # Compute embedding row base pointers
    word_row_ptr = word_emb_ptr + word_id * hidden_size
    token_type_row_ptr = token_type_emb_ptr + token_type_id * hidden_size
    position_row_ptr = position_emb_ptr + position_id * hidden_size

    # Output row pointers (flattened [batch*seq_len, hidden_size] layout)
    output_sum_row_ptr = output_sum_ptr + pid * hidden_size
    output_ln_row_ptr = output_ln_ptr + pid * hidden_size

    # Accumulators for mean/variance
    total_sum = 0.0
    total_sum_sq = 0.0

    # First pass: compute embedding sum, store, accumulate statistics
    for chunk_start in range(0, hidden_size, BLOCK_SIZE):
        chunk_offsets = chunk_start + tl.arange(0, BLOCK_SIZE)
        chunk_mask = chunk_offsets < hidden_size

        # Load word embedding with padding_idx=0 handling
        if word_id == 0:
            word_chunk = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        else:
            word_chunk = tl.load(word_row_ptr + chunk_offsets, mask=chunk_mask, other=0.0).to(tl.float32)

        token_type_chunk = tl.load(token_type_row_ptr + chunk_offsets, mask=chunk_mask, other=0.0).to(tl.float32)
        position_chunk = tl.load(position_row_ptr + chunk_offsets, mask=chunk_mask, other=0.0).to(tl.float32)

        # Sum three embeddings
        sum_chunk = word_chunk + token_type_chunk + position_chunk

        # Store sum to buffer (needed for second pass and possibly as output)
        tl.store(output_sum_row_ptr + chunk_offsets, sum_chunk, mask=chunk_mask)

        # Accumulate statistics with proper masking
        masked_sum = tl.where(chunk_mask, sum_chunk, 0.0)
        total_sum += tl.sum(masked_sum)
        masked_sum_sq = tl.where(chunk_mask, sum_chunk * sum_chunk, 0.0)
        total_sum_sq += tl.sum(masked_sum_sq)

    # Compute mean and variance
    mean = total_sum / hidden_size
    variance = total_sum_sq / hidden_size - mean * mean
    rstd = 1.0 / tl.sqrt(variance + eps_val)

    # Second pass: load sum, normalize, store LN output
    for chunk_start in range(0, hidden_size, BLOCK_SIZE):
        chunk_offsets = chunk_start + tl.arange(0, BLOCK_SIZE)
        chunk_mask = chunk_offsets < hidden_size

        # Load sum from buffer
        sum_chunk = tl.load(output_sum_row_ptr + chunk_offsets, mask=chunk_mask, other=0.0).to(tl.float32)

        # Load LN parameters
        ln_w = tl.load(ln_weight_ptr + chunk_offsets, mask=chunk_mask, other=1.0).to(tl.float32)
        ln_b = tl.load(ln_bias_ptr + chunk_offsets, mask=chunk_mask, other=0.0).to(tl.float32)

        # Normalize: (x - mean) * rstd * weight + bias
        normalized = (sum_chunk - mean) * rstd
        output_chunk = normalized * ln_w + ln_b

        # Store LN output
        tl.store(output_ln_row_ptr + chunk_offsets, output_chunk, mask=chunk_mask)


# ========== Dispatch Wrapper ==========

@torch.fx.wrap
def fused_emb_sum_ln_dispatch(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, route_str):
    # Route: A_both_1024 - Ordering A, hidden=1024, returns both outputs
    if route_str == "A_both_1024":
        input_ids = in_0
        position_emb = in_1
        token_type_emb = in_2
        word_emb = in_3
        ln_bias = in_4
        ln_weight = in_5
        token_type_ids = in_6
        position_ids = in_7
        hidden_size_val = 1024
        return_both = True
    # Route: A_both_768 - Ordering A, hidden=768, returns both outputs
    elif route_str == "A_both_768":
        input_ids = in_0
        position_emb = in_1
        token_type_emb = in_2
        word_emb = in_3
        ln_bias = in_4
        ln_weight = in_5
        token_type_ids = in_6
        position_ids = in_7
        hidden_size_val = 768
        return_both = True
    # Route: A_both_64 - Ordering A, hidden=64, returns both outputs
    elif route_str == "A_both_64":
        input_ids = in_0
        position_emb = in_1
        token_type_emb = in_2
        word_emb = in_3
        ln_bias = in_4
        ln_weight = in_5
        token_type_ids = in_6
        position_ids = in_7
        hidden_size_val = 64
        return_both = True
    # Route: B_ln_768 - Ordering B, hidden=768, returns LN only
    elif route_str == "B_ln_768":
        input_ids = in_0
        ln_bias = in_1
        ln_weight = in_2
        position_emb = in_3
        token_type_emb = in_4
        word_emb = in_5
        token_type_ids = in_6
        position_ids = in_7
        hidden_size_val = 768
        return_both = False
    # Route: B_ln_32 - Ordering B, hidden=32, returns LN only
    elif route_str == "B_ln_32":
        input_ids = in_0
        ln_bias = in_1
        ln_weight = in_2
        position_emb = in_3
        token_type_emb = in_4
        word_emb = in_5
        token_type_ids = in_6
        position_ids = in_7
        hidden_size_val = 32
        return_both = False
    else:
        raise ValueError(f"Unknown route: {route_str}")

    # Compute dimensions
    batch_size = input_ids.shape[0]
    seq_len_val = input_ids.shape[1]

    # Compute BLOCK_SIZE (next power of 2 >= hidden_size, minimum 32)
    bs = 32
    while bs < hidden_size_val:
        bs *= 2
    block_size_val = bs

    # Allocate output tensors
    output_sum = torch.empty(batch_size, seq_len_val, hidden_size_val, dtype=word_emb.dtype, device=word_emb.device)
    output_ln = torch.empty(batch_size, seq_len_val, hidden_size_val, dtype=word_emb.dtype, device=word_emb.device)

    # Compute strides
    input_ids_s0 = input_ids.stride(0)
    position_ids_s0 = position_ids.stride(0)

    # Launch kernel
    num_rows = batch_size * seq_len_val
    num_warps = 4 if block_size_val <= 128 else 8

    fused_emb_sum_ln_kernel[(num_rows,)](
        input_ids_ptr=input_ids,
        token_type_ids_ptr=token_type_ids,
        position_ids_ptr=position_ids,
        word_emb_ptr=word_emb,
        token_type_emb_ptr=token_type_emb,
        position_emb_ptr=position_emb,
        ln_weight_ptr=ln_weight,
        ln_bias_ptr=ln_bias,
        output_sum_ptr=output_sum,
        output_ln_ptr=output_ln,
        seq_len=seq_len_val,
        hidden_size=hidden_size_val,
        input_ids_stride0=input_ids_s0,
        position_ids_stride0=position_ids_s0,
        eps_val=1e-12,
        RETURN_BOTH=return_both,
        BLOCK_SIZE=block_size_val,
        num_warps=num_warps,
    )

    if return_both:
        return (output_sum, output_ln)
    else:
        return (output_ln,)


# ========== Pattern Function (Ordering A, hidden=768, returns both) ==========

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    tmp_7 = torch.nn.functional.embedding(in_0, in_3, 0, None, 2.0, False, False)
    tmp_8 = torch.nn.functional.embedding(in_6, in_2, None, None, 2.0, False, False)
    tmp_9 = tmp_7 + tmp_8
    tmp_10 = torch.nn.functional.embedding(in_7, in_1, None, None, 2.0, False, False)
    tmp_9 += tmp_10
    tmp_12 = torch.nn.functional.dropout(tmp_9, 0.1, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (768,), in_5, in_4, 1e-12)
    return (tmp_12, tmp_13)


# ========== Replacement Args ==========

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, "A_both_768")


# ========== Replacement Func ==========

def replacement_func():
    return fused_emb_sum_ln_dispatch