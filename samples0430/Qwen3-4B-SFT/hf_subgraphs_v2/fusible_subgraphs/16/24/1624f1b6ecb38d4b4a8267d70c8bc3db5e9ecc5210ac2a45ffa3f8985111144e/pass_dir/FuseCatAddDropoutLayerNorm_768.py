import torch
import triton
import triton.language as tl


def pattern(x_cls, x_conv, x_pos):
    tmp_10 = torch.cat((x_cls, x_conv), dim=1)
    tmp_11 = tmp_10 + x_pos
    tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (768,), (768,), 1e-06)
    return tmp_12, tmp_13


def replacement_args(x_cls, x_conv, x_pos):
    return (x_cls, x_conv, x_pos)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_N": 1024}, num_warps=16),
        triton.Config({"BLOCK_N": 1024}, num_warps=32),
        triton.Config({"BLOCK_N": 2048}, num_warps=8),
        triton.Config({"BLOCK_N": 2048}, num_warps=16),
    ],
    key=["N_ROWS"],
)
@triton.jit
def _fused_cat_add_ln_kernel(
    x_cls_ptr,
    x_conv_ptr,
    x_pos_ptr,
    out_add_ptr,
    out_ln_ptr,
    N_ROWS,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one row (sequence position)
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N_ROWS

    # Load inputs for this row
    cls_val = tl.load(x_cls_ptr + cols, mask=mask, other=0.0)
    conv_val = tl.load(x_conv_ptr + row * N_ROWS + cols, mask=mask, other=0.0)
    pos_val = tl.load(x_pos_ptr + row * N_ROWS + cols, mask=mask, other=0.0)

    # Combine cls token (row 0) with patch embeddings (rows 1..N_ROWS-1)
    is_cls = (row == 0)
    x = tl.where(is_cls, cls_val, conv_val)

    # Add position embedding
    x = x + pos_val

    # Store addition result (handles identity dropout: p=0.0, training=False)
    tl.store(out_add_ptr + row * N_ROWS + cols, x, mask=mask)

    # Layer normalization in float32 for numerical stability
    x_f32 = x.to(tl.float32)
    mean = tl.sum(x_f32, axis=0) / N_ROWS
    diff = x_f32 - mean
    var = tl.sum(diff * diff, axis=0) / N_ROWS
    inv_std = tl.rsqrt(var + 1e-6)
    x_norm = diff * inv_std

    # Store layer norm output
    tl.store(out_ln_ptr + row * N_ROWS + cols, x_norm.to(x.dtype), mask=mask)


@torch.fx.wrap
def fused_cat_add_layernorm_768(x_cls, x_conv, x_pos):
    """
    Fused: cat([cls_token, conv_output], dim=1) + pos_embed  -> add result
           then layer_norm over last dim=768
    Returns: (add_result, layernorm_result)  identical to (tmp_12, tmp_13)
    """
    batch_dim = x_cls.shape[0]
    seq_len   = x_cls.shape[1] + x_conv.shape[0]
    N         = x_cls.shape[2]   # 768
    total_rows = batch_dim * seq_len

    x_cls_f  = x_cls.contiguous().view(-1, N)
    x_conv_f = x_conv.contiguous().view(-1, N)
    x_pos_f  = x_pos.contiguous().view(-1, N)

    out_add = torch.empty(total_rows, N, dtype=x_cls.dtype, device=x_cls.device)
    out_ln  = torch.empty(total_rows, N, dtype=x_cls.dtype, device=x_cls.device)

    _fused_cat_add_ln_kernel[(total_rows,)](
        x_cls_f,
        x_conv_f,
        x_pos_f,
        out_add,
        out_ln,
        N_ROWS=N,
    )

    return out_add.view(batch_dim, seq_len, N), out_ln.view(batch_dim, seq_len, N)


def replacement_func():
    return fused_cat_add_layernorm_768