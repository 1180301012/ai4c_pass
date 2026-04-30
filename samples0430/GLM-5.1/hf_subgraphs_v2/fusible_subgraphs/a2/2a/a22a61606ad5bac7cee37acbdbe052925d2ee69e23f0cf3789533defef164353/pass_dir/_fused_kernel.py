import torch
import triton
import triton.language as tl


@triton.jit
def fused_embed_add_ln_kernel(
    input_ids_ptr, token_type_ids_ptr, position_ids_ptr,
    word_emb_ptr, token_type_emb_ptr, position_emb_ptr,
    ln_weight_ptr, ln_bias_ptr,
    sum_buf_ptr, ln_out_ptr,
    n_rows, hidden_size,
    word_emb_stride, token_type_emb_stride, position_emb_stride,
    eps,
    BLOCK_H: tl.constexpr,
    HAS_DROPOUT_OUTPUT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    # Get embedding indices for this position
    input_id = tl.load(input_ids_ptr + row_idx).to(tl.int32)
    tt_id = tl.load(token_type_ids_ptr + row_idx).to(tl.int32)
    pos_id = tl.load(position_ids_ptr + row_idx).to(tl.int32)

    # Compute base pointers for embedding rows
    word_row_ptr = word_emb_ptr + input_id * word_emb_stride
    tt_row_ptr = token_type_emb_ptr + tt_id * token_type_emb_stride
    pos_row_ptr = position_emb_ptr + pos_id * position_emb_stride

    # Handle padding_idx=0 for word embeddings
    is_pad = input_id == 0

    # Phase 1: Compute sum, mean, and variance
    sum_acc = 0.0
    sum_sq_acc = 0.0

    for h_start in range(0, hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hidden_size

        # Load embedding values, cast to float32 for computation
        if is_pad:
            word_vals = tl.zeros([BLOCK_H], dtype=tl.float32)
        else:
            word_vals = tl.load(word_row_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
        tt_vals = tl.load(tt_row_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
        pos_vals = tl.load(pos_row_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)

        # Sum embeddings
        embed_sum = word_vals + tt_vals + pos_vals

        # Accumulate for mean and variance
        sum_acc += tl.sum(embed_sum, axis=0)
        sum_sq_acc += tl.sum(embed_sum * embed_sum, axis=0)

        # Store sum to buffer (auto-cast to buffer's dtype)
        tl.store(sum_buf_ptr + row_idx * hidden_size + h_offsets, embed_sum, mask=h_mask)

    # Compute mean and variance
    mean = sum_acc / hidden_size
    var = sum_sq_acc / hidden_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Phase 2: Read sum from buffer, normalize, apply gamma/beta
    for h_start in range(0, hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hidden_size

        # Read sum from buffer, cast to float32
        embed_sum = tl.load(sum_buf_ptr + row_idx * hidden_size + h_offsets, mask=h_mask, other=0.0).to(tl.float32)

        # Normalize
        normalized = (embed_sum - mean) * rstd

        # Load gamma and beta
        gamma = tl.load(ln_weight_ptr + h_offsets, mask=h_mask, other=1.0).to(tl.float32)
        beta = tl.load(ln_bias_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)

        # Apply gamma * normalized + beta
        output = gamma * normalized + beta

        # Store LN output (auto-cast to output dtype)
        tl.store(ln_out_ptr + row_idx * hidden_size + h_offsets, output, mask=h_mask)


@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    input_ids = args[0]
    word_emb = args[1]
    token_type_ids = args[2]
    token_type_emb = args[3]
    position_ids = args[4]
    position_emb = args[5]
    ln_weight = args[6]
    ln_bias = args[7]

    hidden_size = ln_weight.shape[0]
    eps = 1e-12
    out_dtype = word_emb.dtype
    device = word_emb.device

    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1] if input_ids.dim() > 1 else 1
    n_rows = batch_size * seq_len

    is_2out = route.endswith("_2out")

    # Allocate output buffers
    if is_2out:
        # For 2-value return, sum_buf IS the dropout output (original dtype)
        sum_buf = torch.empty(n_rows, hidden_size, dtype=out_dtype, device=device)
    else:
        # For 1-value return, sum_buf is a temporary float32 buffer
        sum_buf = torch.empty(n_rows, hidden_size, dtype=torch.float32, device=device)

    ln_out = torch.empty(n_rows, hidden_size, dtype=out_dtype, device=device)

    # Flatten and broadcast index tensors
    input_ids_flat = input_ids.reshape(-1).contiguous()
    token_type_ids_flat = token_type_ids.reshape(-1).contiguous()
    # position_ids may have shape [1, seq_len], need to broadcast to [batch, seq_len]
    if position_ids.shape[0] == 1 and batch_size > 1:
        position_ids_flat = position_ids.expand(batch_size, -1).reshape(-1).contiguous()
    else:
        position_ids_flat = position_ids.reshape(-1).contiguous()

    # Make embedding tables contiguous
    word_emb_c = word_emb.contiguous()
    token_type_emb_c = token_type_emb.contiguous()
    position_emb_c = position_emb.contiguous()
    ln_weight_c = ln_weight.contiguous()
    ln_bias_c = ln_bias.contiguous()

    # Choose BLOCK_H based on hidden_size
    # Use a power-of-2 block size that balances loop iterations and register usage
    if hidden_size <= 64:
        BLOCK_H = 64
    elif hidden_size <= 256:
        BLOCK_H = 128
    elif hidden_size <= 512:
        BLOCK_H = 256
    else:
        BLOCK_H = 256

    grid = (n_rows,)

    fused_embed_add_ln_kernel[grid](
        input_ids_flat, token_type_ids_flat, position_ids_flat,
        word_emb_c, token_type_emb_c, position_emb_c,
        ln_weight_c, ln_bias_c,
        sum_buf, ln_out,
        n_rows, hidden_size,
        word_emb_c.shape[1], token_type_emb_c.shape[1], position_emb_c.shape[1],
        eps,
        BLOCK_H=BLOCK_H,
        HAS_DROPOUT_OUTPUT=is_2out,
    )

    # Reshape outputs to 3D
    ln_out_3d = ln_out.reshape(batch_size, seq_len, hidden_size)

    if is_2out:
        dropout_out_3d = sum_buf.reshape(batch_size, seq_len, hidden_size)
        return (dropout_out_3d, ln_out_3d)
    else:
        return (ln_out_3d,)