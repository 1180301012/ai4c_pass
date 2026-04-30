import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv_ln_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, ln_weight_ptr, ln_bias_ptr, output_ptr,
    batch_size, c_in, c_out,
    input_stride_n, input_stride_c,
    weight_stride_oc, weight_stride_ic,
    ln_w_stride_0, ln_b_stride_0,
    output_stride_n, output_stride_c,
    eps,
    BLOCK_C_IN: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    DTYPE_ID: tl.constexpr,  # 0=float32, 1=float16, 2=bfloat16
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    c_out_offsets = tl.arange(0, BLOCK_C_OUT)
    c_out_mask = c_out_offsets < c_out
    mask_float = c_out_mask.to(tl.float32)

    # Load conv bias [C_out] and cast to float32
    bias_vals = tl.load(bias_ptr + c_out_offsets, mask=c_out_mask, other=0.0).to(tl.float32)

    # Load LN weight [C_out, 1, 1] and LN bias [C_out, 1, 1] - effectively 1D
    ln_weight_vals = tl.load(ln_weight_ptr + c_out_offsets * ln_w_stride_0, mask=c_out_mask, other=1.0).to(tl.float32)
    ln_bias_vals = tl.load(ln_bias_ptr + c_out_offsets * ln_b_stride_0, mask=c_out_mask, other=0.0).to(tl.float32)

    # Initialize conv output with bias
    conv_out = bias_vals  # [BLOCK_C_OUT], float32

    # Accumulate dot product across input channels in tiles
    for c_in_start in range(0, c_in, BLOCK_C_IN):
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_C_IN)
        c_in_mask = c_in_offsets < c_in

        # Load input for this batch element: input[pid, c_in_offsets, 0, 0]
        input_offsets = pid * input_stride_n + c_in_offsets * input_stride_c
        input_vals = tl.load(input_ptr + input_offsets, mask=c_in_mask, other=0.0).to(tl.float32)

        # Load weight slice: weight[c_out_offsets, c_in_offsets, 0, 0]
        weight_offsets = c_out_offsets[:, None] * weight_stride_oc + c_in_offsets[None, :] * weight_stride_ic
        weight_mask_2d = c_out_mask[:, None] & c_in_mask[None, :]
        weight_vals = tl.load(weight_ptr + weight_offsets, mask=weight_mask_2d, other=0.0).to(tl.float32)

        # Dot product: sum over c_in dimension
        conv_out += tl.sum(weight_vals * input_vals[None, :], axis=1)

    # Compute LayerNorm across C_out channels for this batch element
    # Mean
    mean = tl.sum(conv_out * mask_float, axis=0) / c_out
    # Difference from mean (zero out invalid channels)
    diff = (conv_out - mean) * mask_float
    # Variance
    var = tl.sum(diff * diff, axis=0) / c_out
    # Reciprocal of standard deviation
    rstd = 1.0 / tl.sqrt(var + eps)
    # Normalize
    normalized = diff * rstd
    # Scale and shift with LN parameters
    ln_out = normalized * ln_weight_vals + ln_bias_vals
    # ReLU
    result = tl.maximum(ln_out, 0.0) * mask_float

    # Cast result to output dtype
    if DTYPE_ID == 0:
        result_store = result
    elif DTYPE_ID == 1:
        result_store = result.to(tl.float16)
    else:  # DTYPE_ID == 2, bfloat16
        result_store = result.to(tl.bfloat16)

    # Store result: output[pid, c_out_offsets, 0, 0]
    output_offsets = pid * output_stride_n + c_out_offsets * output_stride_c
    tl.store(output_ptr + output_offsets, result_store, mask=c_out_mask)


@torch.fx.wrap
def fused_conv_ln_relu_dispatch(conv_bias, conv_weight, ln_bias, ln_weight, input_tensor, route):
    """Shared dispatch wrapper for all fused conv-ln-relu passes."""
    # Route determines the normalized_shape, but we extract C_out from weight
    batch_size = input_tensor.shape[0]
    c_in = input_tensor.shape[1]
    c_out = conv_weight.shape[0]

    # Create output tensor with same dtype and device as input
    output = torch.empty((batch_size, c_out, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)

    # Get strides
    input_stride_n = input_tensor.stride(0)
    input_stride_c = input_tensor.stride(1)
    weight_stride_oc = conv_weight.stride(0)
    weight_stride_ic = conv_weight.stride(1)
    ln_w_stride_0 = ln_weight.stride(0)
    ln_b_stride_0 = ln_bias.stride(0)
    output_stride_n = output.stride(0)
    output_stride_c = output.stride(1)

    # Determine dtype ID
    if input_tensor.dtype == torch.float32:
        dtype_id = 0
    elif input_tensor.dtype == torch.float16:
        dtype_id = 1
    elif input_tensor.dtype == torch.bfloat16:
        dtype_id = 2
    else:
        raise ValueError(f"Unsupported dtype: {input_tensor.dtype}")

    # Choose BLOCK_C_OUT based on c_out (next power of 2, but at least 16)
    if c_out <= 16:
        BLOCK_C_OUT = 16
    elif c_out <= 32:
        BLOCK_C_OUT = 32
    elif c_out <= 64:
        BLOCK_C_OUT = 64
    else:
        BLOCK_C_OUT = 128

    BLOCK_C_IN = 32

    # Grid: one program per batch element
    grid = (batch_size,)

    # Launch kernel
    fused_conv_ln_relu_kernel[grid](
        input_ptr=input_tensor, weight_ptr=conv_weight, bias_ptr=conv_bias,
        ln_weight_ptr=ln_weight, ln_bias_ptr=ln_bias, output_ptr=output,
        batch_size=batch_size, c_in=c_in, c_out=c_out,
        input_stride_n=input_stride_n, input_stride_c=input_stride_c,
        weight_stride_oc=weight_stride_oc, weight_stride_ic=weight_stride_ic,
        ln_w_stride_0=ln_w_stride_0, ln_b_stride_0=ln_b_stride_0,
        output_stride_n=output_stride_n, output_stride_c=output_stride_c,
        eps=1e-05,
        BLOCK_C_IN=BLOCK_C_IN,
        BLOCK_C_OUT=BLOCK_C_OUT,
        DTYPE_ID=dtype_id,
    )

    return output