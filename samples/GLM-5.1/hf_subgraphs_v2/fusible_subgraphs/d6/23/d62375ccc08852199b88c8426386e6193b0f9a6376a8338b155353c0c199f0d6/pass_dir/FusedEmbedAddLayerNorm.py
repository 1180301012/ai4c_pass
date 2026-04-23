import torch
import triton
import triton.language as tl

# ===== Triton kernels =====

@triton.jit
def layernorm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    hidden_size,
    eps,
    BLOCK_H: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * hidden_size

    # Pass 1: compute mean
    total_sum = 0.0
    for h_start in range(0, hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask = h_offsets < hidden_size
        x_vals = tl.load(x_ptr + row_start + h_offsets, mask=mask, other=0.0).to(tl.float32)
        total_sum += tl.sum(x_vals)
    mean = total_sum / hidden_size

    # Pass 2: compute variance
    total_var = 0.0
    for h_start in range(0, hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask = h_offsets < hidden_size
        x_vals = tl.load(x_ptr + row_start + h_offsets, mask=mask, other=0.0).to(tl.float32)
        diff = x_vals - mean
        total_var += tl.sum(diff * diff)
    var = total_var / hidden_size
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 3: normalize and apply weight/bias
    for h_start in range(0, hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask = h_offsets < hidden_size
        x_vals = tl.load(x_ptr + row_start + h_offsets, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(w_ptr + h_offsets, mask=mask, other=0.0).to(tl.float32)
        b_vals = tl.load(b_ptr + h_offsets, mask=mask, other=0.0).to(tl.float32)

        normalized = (x_vals - mean) * rstd
        out = normalized * w_vals + b_vals

        tl.store(out_ptr + row_start + h_offsets, out.to(x_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def triton_layernorm_1024(x, w, b):
    hidden_size = 1024
    out = torch.empty_like(x)
    rows = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
    BLOCK_H = 1024
    grid = (rows,)
    layernorm_kernel[grid](
        x_ptr=x,
        w_ptr=w,
        b_ptr=b,
        out_ptr=out,
        hidden_size=hidden_size,
        eps=1e-05,
        BLOCK_H=BLOCK_H,
    )
    return out

@torch.fx.wrap
def triton_layernorm_2048(x, w, b):
    hidden_size = 2048
    out = torch.empty_like(x)
    rows = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
    BLOCK_H = 2048
    grid = (rows,)
    layernorm_kernel[grid](
        x_ptr=x,
        w_ptr=w,
        b_ptr=b,
        out_ptr=out,
        hidden_size=hidden_size,
        eps=1e-05,
        BLOCK_H=BLOCK_H,
    )
    return out

@torch.fx.wrap
def dispatch_fused_layernorm(x, w, b, route):
    if route == "route_ln_1024":
        return triton_layernorm_1024(x, w, b)
    elif route == "route_ln_2048":
        return triton_layernorm_2048(x, w, b)
    else:
        raise RuntimeError(f"Unknown route: {route}")

@torch.fx.wrap
def fused_embed_add_layernorm_1024(input_embed, pos_weight, ln_bias, ln_weight, seq_len_val, hidden_size_val, offset_val, eps_val):
    seq_len = seq_len_val
    hidden_size = hidden_size_val

    add_out = torch.empty_like(input_embed)
    ln_out = torch.empty_like(input_embed)

    BLOCK_H = 1024

    grid = (seq_len,)

    fused_embed_add_layernorm_kernel[grid](
        input_embed_ptr=input_embed,
        pos_weight_ptr=pos_weight,
        ln_weight_ptr=ln_weight,
        ln_bias_ptr=ln_bias,
        add_out_ptr=add_out,
        ln_out_ptr=ln_out,
        seq_len=seq_len,
        hidden_size=hidden_size,
        offset=offset_val,
        eps=eps_val,
        BLOCK_H=BLOCK_H,
    )

    return add_out, ln_out


@torch.fx.wrap
def fused_embed_add_layernorm_2048(input_embed, pos_weight, ln_bias, ln_weight, seq_len_val, hidden_size_val, offset_val, eps_val):
    seq_len = seq_len_val
    hidden_size = hidden_size_val

    add_out = torch.empty_like(input_embed)
    ln_out = torch.empty_like(input_embed)

    BLOCK_H = 2048

    grid = (seq_len,)

    fused_embed_add_layernorm_kernel[grid](
        input_embed_ptr=input_embed,
        pos_weight_ptr=pos_weight,
        ln_weight_ptr=ln_weight,
        ln_bias_ptr=ln_bias,
        add_out_ptr=add_out,
        ln_out_ptr=ln_out,
        seq_len=seq_len,
        hidden_size=hidden_size,
        offset=offset_val,
        eps=eps_val,
        BLOCK_H=BLOCK_H,
    )

    return add_out, ln_out


# ===== Shared dispatch wrapper =====

@torch.fx.wrap
def dispatch_fused_embed_add_layernorm(input_embed, pos_weight, ln_bias, ln_weight, seq_len_val, hidden_size_val, offset_val, eps_val, route):
    if route == "route_1024":
        return fused_embed_add_layernorm_1024(input_embed, pos_weight, ln_bias, ln_weight, seq_len_val, hidden_size_val, offset_val, eps_val)
    elif route == "route_2048":
        return fused_embed_add_layernorm_2048(input_embed, pos_weight, ln_bias, ln_weight, seq_len_val, hidden_size_val, offset_val, eps_val)
    else:
        raise RuntimeError(f"Unknown route: {route}")

def replacement_func():
    return dispatch_fused_embed_add_layernorm