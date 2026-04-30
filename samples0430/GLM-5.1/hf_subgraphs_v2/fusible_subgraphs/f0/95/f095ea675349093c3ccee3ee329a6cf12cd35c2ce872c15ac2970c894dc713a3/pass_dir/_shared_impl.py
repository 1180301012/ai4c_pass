import torch
import triton
import triton.language as tl

# Triton kernel for fused: slice + gelu + transpose + add + layer_norm
# Since dropout with training=False is identity, we skip it entirely
# Input: conv1d_output (B, C, T_in) where T_in = T_out + 1 due to slice [:,:,:-1]
# The slice and transpose are implicit: we read conv[B, c, t] for t < T_out and add to hidden[B, t, c]
@triton.jit
def fused_gelu_trans_add_layernorm_kernel(
    conv_ptr, hidden_ptr, ln_weight_ptr, ln_bias_ptr,
    add_out_ptr, ln_out_ptr,
    B, T_out, C,
    stride_conv_b, stride_conv_c, stride_conv_t,
    stride_hidden_b, stride_hidden_t, stride_hidden_c,
    stride_add_out_b, stride_add_out_t, stride_add_out_c,
    stride_ln_out_b, stride_ln_out_t, stride_ln_out_c,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(1)
    time_idx = tl.program_id(0)
    
    if batch_idx >= B or time_idx >= T_out:
        return
    
    channel_offsets = tl.arange(0, BLOCK_SIZE)
    mask = channel_offsets < C
    
    # Load conv1d output at [b, c, t] where t < T_out (implicit slice [:,:,:-1])
    # The transpose is implicit: we read conv[B, C, T] and add to hidden[B, T, C]
    conv_offsets = batch_idx * stride_conv_b + channel_offsets * stride_conv_c + time_idx * stride_conv_t
    conv_vals = tl.load(conv_ptr + conv_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Apply gelu: gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt2 = 1.4142135623730951
    gelu_vals = conv_vals * 0.5 * (1.0 + tl.math.erf(conv_vals / sqrt2))
    
    # Load hidden states at [b, t, c]
    hidden_offsets = batch_idx * stride_hidden_b + time_idx * stride_hidden_t + channel_offsets * stride_hidden_c
    hidden_vals = tl.load(hidden_ptr + hidden_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Add (transpose + add fusion)
    add_vals = hidden_vals + gelu_vals
    
    # Compute layer_norm statistics in float32 for precision
    # Using parallel variance formula: var = E[x^2] - E[x]^2
    sum_val = tl.sum(add_vals, axis=0)
    sum_sq_val = tl.sum(add_vals * add_vals, axis=0)
    
    mean = sum_val / C
    variance = sum_sq_val / C - mean * mean
    rstd = 1.0 / tl.sqrt(variance + 1e-5)
    
    # Normalize
    ln_vals = (add_vals - mean) * rstd
    
    # Load weight and bias for layer_norm affine transform
    ln_weight_vals = tl.load(ln_weight_ptr + channel_offsets, mask=mask, other=1.0).to(tl.float32)
    ln_bias_vals = tl.load(ln_bias_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
    
    ln_vals = ln_vals * ln_weight_vals + ln_bias_vals
    
    # Store add output (same as dropout output since dropout is identity in eval)
    add_out_offsets = batch_idx * stride_add_out_b + time_idx * stride_add_out_t + channel_offsets * stride_add_out_c
    tl.store(add_out_ptr + add_out_offsets, add_vals, mask=mask)
    
    # Store layer_norm output
    ln_out_offsets = batch_idx * stride_ln_out_b + time_idx * stride_ln_out_t + channel_offsets * stride_ln_out_c
    tl.store(ln_out_ptr + ln_out_offsets, ln_vals, mask=mask)


# Internal implementation function
# Uses only allowed APIs: torch.empty_like, torch.empty, tensor.shape, tensor.stride()
def _fused_gelu_trans_add_layernorm(conv_output, hidden_states, ln_weight, ln_bias):
    # conv_output shape: (B, C, T_in) where T_in = T_out + 1 (before slice [:,:,:-1])
    # hidden_states shape: (B, T_out, C)
    B = conv_output.shape[0]
    C = conv_output.shape[1]
    T_in = conv_output.shape[2]
    T_out = T_in - 1  # Slice removes last time step
    
    # Allocate output tensors using allowed APIs
    add_output = torch.empty_like(hidden_states)  # shape (B, T_out, C)
    ln_output = torch.empty_like(hidden_states)    # shape (B, T_out, C)
    
    # Get strides for memory access
    stride_conv_b, stride_conv_c, stride_conv_t = conv_output.stride()
    stride_hidden_b, stride_hidden_t, stride_hidden_c = hidden_states.stride()
    stride_add_out_b, stride_add_out_t, stride_add_out_c = add_output.stride()
    stride_ln_out_b, stride_ln_out_t, stride_ln_out_c = ln_output.stride()
    
    # BLOCK_SIZE must be >= C to handle all channels in one pass
    BLOCK_SIZE = triton.next_power_of_2(C)
    
    # Grid: each program handles one (batch, time_step) pair
    grid = (T_out, B)
    
    fused_gelu_trans_add_layernorm_kernel[grid](
        conv_output, hidden_states, ln_weight, ln_bias,
        add_output, ln_output,
        B, T_out, C,
        stride_conv_b, stride_conv_c, stride_conv_t,
        stride_hidden_b, stride_hidden_t, stride_hidden_c,
        stride_add_out_b, stride_add_out_t, stride_add_out_c,
        stride_ln_out_b, stride_ln_out_t, stride_ln_out_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return add_output, ln_output


# Shared dispatch wrapper - handles all route strings
# This is the single function object that all passes share via replacement_func()
@torch.fx.wrap
def fused_gelu_trans_add_layernorm_dispatch(conv_output, hidden_states, ln_weight, ln_bias, route):
    if route == "p005":
        return _fused_gelu_trans_add_layernorm(conv_output, hidden_states, ln_weight, ln_bias)
    elif route == "p010":
        return _fused_gelu_trans_add_layernorm(conv_output, hidden_states, ln_weight, ln_bias)
    else:
        raise ValueError(f"Unknown route: {route}")