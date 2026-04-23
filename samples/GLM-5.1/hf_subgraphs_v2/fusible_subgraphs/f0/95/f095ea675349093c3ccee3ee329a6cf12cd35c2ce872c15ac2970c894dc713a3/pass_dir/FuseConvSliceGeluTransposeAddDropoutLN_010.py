import torch
import triton
import triton.language as tl

# Pattern matching function - matches float16/float32 model with dropout p=0.1
# Matches the subgraph AFTER conv1d: slice, gelu, transpose, add, dropout(identity), layer_norm
def pattern(conv_out, residual, ln_weight, ln_bias):
    sliced = conv_out[:, :, :-1]
    gelu_out = torch.nn.functional.gelu(sliced)
    transposed = gelu_out.transpose(1, 2)
    added = residual + transposed
    dropped = torch.nn.functional.dropout(added, 0.1, False, False)
    ln_out = torch.nn.functional.layer_norm(dropped, (1024,), ln_weight, ln_bias, 1e-05)
    return (dropped, ln_out)

# Argument extraction function - with route string as last arg
def replacement_args(conv_out, residual, ln_weight, ln_bias):
    return (conv_out, residual, ln_weight, ln_bias, "route_010")

# Triton kernel: fused slice + gelu + transpose + add + identity_dropout + layer_norm
@triton.jit
def fused_slice_gelu_trans_add_ln_kernel(
    conv_out_ptr,
    residual_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    dropout_out_ptr,
    ln_out_ptr,
    C_out,
    L_sliced,
    stride_conv_b, stride_conv_c, stride_conv_l,
    stride_res_b, stride_res_l, stride_res_c,
    stride_ln_w,
    stride_ln_b,
    stride_drop_b, stride_drop_l, stride_drop_c,
    stride_ln_out_b, stride_ln_out_l, stride_ln_out_c,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    batch_idx = row_idx // L_sliced
    seq_idx = row_idx % L_sliced

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < C_out

    # Read conv_out at position (batch_idx, channel, seq_idx) for all channels
    conv_offsets = batch_idx * stride_conv_b + offsets * stride_conv_c + seq_idx * stride_conv_l
    conv_vals = tl.load(conv_out_ptr + conv_offsets, mask=mask, other=0.0)

    # Apply GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt2 = 1.4142135623730951
    gelu_vals = conv_vals * 0.5 * (1.0 + tl.math.erf(conv_vals / sqrt2))

    # Read residual at position (batch_idx, seq_idx, channel) for all channels
    res_offsets = batch_idx * stride_res_b + seq_idx * stride_res_l + offsets * stride_res_c
    res_vals = tl.load(residual_ptr + res_offsets, mask=mask, other=0.0)

    # Add: residual + gelu(transposed conv slice)
    add_vals = res_vals + gelu_vals

    # Dropout with training=False is identity (no scaling)

    # Compute mean for layer norm
    mean_val = tl.sum(add_vals, axis=0) / C_out

    # Compute variance for layer norm
    diff = add_vals - mean_val
    var_val = tl.sum(diff * diff, axis=0) / C_out

    # Layer norm: (x - mean) / sqrt(var + eps) * weight + bias
    ln_weight_vals = tl.load(ln_weight_ptr + offsets * stride_ln_w, mask=mask, other=1.0)
    ln_bias_vals = tl.load(ln_bias_ptr + offsets * stride_ln_b, mask=mask, other=0.0)

    rstd = 1.0 / tl.sqrt(var_val + eps)
    normalized = (add_vals - mean_val) * rstd
    ln_vals = normalized * ln_weight_vals + ln_bias_vals

    # Write dropout output = add_vals (identity dropout)
    drop_offsets = batch_idx * stride_drop_b + seq_idx * stride_drop_l + offsets * stride_drop_c
    tl.store(dropout_out_ptr + drop_offsets, add_vals, mask=mask)

    # Write layer norm output
    ln_out_offsets = batch_idx * stride_ln_out_b + seq_idx * stride_ln_out_l + offsets * stride_ln_out_c
    tl.store(ln_out_ptr + ln_out_offsets, ln_vals, mask=mask)


@torch.fx.wrap
def fused_dispatch(conv_out, residual, ln_weight, ln_bias, route):
    B = conv_out.shape[0]
    C_out = conv_out.shape[1]
    L_out = conv_out.shape[2]
    L_sliced = L_out - 1

    eps = 1e-05
    BLOCK_SIZE = triton.next_power_of_2(C_out)

    # Allocate outputs
    dropout_out = torch.empty((B, L_sliced, C_out), dtype=residual.dtype, device=residual.device)
    ln_out = torch.empty((B, L_sliced, C_out), dtype=residual.dtype, device=residual.device)

    # Compute strides
    sc_b, sc_c, sc_l = conv_out.stride()
    sr_b, sr_l, sr_c = residual.stride()
    slw = ln_weight.stride(0)
    slb = ln_bias.stride(0)
    sd_b, sd_l, sd_c = dropout_out.stride()
    sno_b, sno_l, sno_c = ln_out.stride()

    num_rows = B * L_sliced

    if route == "route_005":
        fused_slice_gelu_trans_add_ln_kernel[(num_rows,)](
            conv_out_ptr=conv_out,
            residual_ptr=residual,
            ln_weight_ptr=ln_weight,
            ln_bias_ptr=ln_bias,
            dropout_out_ptr=dropout_out,
            ln_out_ptr=ln_out,
            C_out=C_out,
            L_sliced=L_sliced,
            stride_conv_b=sc_b, stride_conv_c=sc_c, stride_conv_l=sc_l,
            stride_res_b=sr_b, stride_res_l=sr_l, stride_res_c=sr_c,
            stride_ln_w=slw,
            stride_ln_b=slb,
            stride_drop_b=sd_b, stride_drop_l=sd_l, stride_drop_c=sd_c,
            stride_ln_out_b=sno_b, stride_ln_out_l=sno_l, stride_ln_out_c=sno_c,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif route == "route_010":
        fused_slice_gelu_trans_add_ln_kernel[(num_rows,)](
            conv_out_ptr=conv_out,
            residual_ptr=residual,
            ln_weight_ptr=ln_weight,
            ln_bias_ptr=ln_bias,
            dropout_out_ptr=dropout_out,
            ln_out_ptr=ln_out,
            C_out=C_out,
            L_sliced=L_sliced,
            stride_conv_b=sc_b, stride_conv_c=sc_c, stride_conv_l=sc_l,
            stride_res_b=sr_b, stride_res_l=sr_l, stride_res_c=sr_c,
            stride_ln_w=slw,
            stride_ln_b=slb,
            stride_drop_b=sd_b, stride_drop_l=sd_l, stride_drop_c=sd_c,
            stride_ln_out_b=sno_b, stride_ln_out_l=sno_l, stride_ln_out_c=sno_c,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        raise ValueError(f"Unknown route: {route}")

    return (dropout_out, ln_out)


def replacement_func():
    return fused_dispatch