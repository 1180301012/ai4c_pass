import torch
import triton
import triton.language as tl


# ===== Triton Kernels =====

@triton.jit
def rope_kernel(
    freqs_ptr,
    cos_out_ptr,
    sin_out_ptr,
    half_dim,
    n_rows,
    stride_freqs_row,
    stride_freqs_col,
    stride_cos_row,
    stride_cos_col,
    stride_sin_row,
    stride_sin_col,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < half_dim

    # Load freqs value for this row at each column offset, cast to float32
    freq_val = tl.load(freqs_ptr + row_idx * stride_freqs_row + offsets * stride_freqs_col, mask=mask, other=0.0).to(tl.float32)

    # Compute cos/sin in float32 (matches PyTorch internal computation)
    cos_val = tl.cos(freq_val)
    sin_val = tl.sin(freq_val)

    # Write first half of output (positions 0..half_dim-1)
    tl.store(cos_out_ptr + row_idx * stride_cos_row + offsets * stride_cos_col, cos_val, mask=mask)
    tl.store(sin_out_ptr + row_idx * stride_sin_row + offsets * stride_sin_col, sin_val, mask=mask)

    # Write second half of output (positions half_dim..2*half_dim-1) - duplicated
    tl.store(cos_out_ptr + row_idx * stride_cos_row + (offsets + half_dim) * stride_cos_col, cos_val, mask=mask)
    tl.store(sin_out_ptr + row_idx * stride_sin_row + (offsets + half_dim) * stride_sin_col, sin_val, mask=mask)


@triton.jit
def rmsnorm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    hidden_dim,
    eps,
    n_rows,
    stride_input_row,
    stride_input_col,
    stride_output_row,
    stride_output_col,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    input_row_ptr = input_ptr + row_idx * stride_input_row
    output_row_ptr = output_ptr + row_idx * stride_output_row

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    # Load input row as float32
    input_row = tl.load(input_row_ptr + offsets * stride_input_col, mask=mask, other=0.0).to(tl.float32)
    # Load weight as float32
    weight_row = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute mean of squares (RMSNorm: no mean subtraction)
    sum_squares = tl.sum(input_row * input_row, axis=0)
    mean_squares = sum_squares / hidden_dim

    # Compute rsqrt(mean_squares + eps)
    rsqrt_val = tl.rsqrt(mean_squares + eps)

    # Normalize and apply weight: result = input * rsqrt * weight
    result = input_row * rsqrt_val * weight_row

    # Store result (Triton auto-casts to output tensor dtype)
    tl.store(output_row_ptr + offsets * stride_output_col, result, mask=mask)


# ===== Dispatch Wrapper =====

@torch.fx.wrap
def fused_rope_rmsnorm_dispatch(in_0, in_1, in_2, route):
    if route == "route_bf16_eps1e6":
        rope_dtype = torch.bfloat16
        norm_dtype = torch.bfloat16
        eps = 1e-06
    elif route == "route_fp32_eps1e5":
        rope_dtype = torch.float32
        norm_dtype = torch.float32
        eps = 1e-05
    else:
        raise ValueError(f"Unknown route: {route}")

    # === RoPE Computation ===
    freqs_shape = in_1.shape
    half_dim = freqs_shape[-1]
    n_rows_rope = in_1.numel() // half_dim
    full_dim = half_dim * 2

    out_shape = list(freqs_shape)
    out_shape[-1] = full_dim
    cos_out = torch.empty(out_shape, dtype=rope_dtype, device=in_1.device)
    sin_out = torch.empty(out_shape, dtype=rope_dtype, device=in_1.device)

    freqs_2d = in_1.reshape(n_rows_rope, half_dim)
    cos_2d = cos_out.reshape(n_rows_rope, full_dim)
    sin_2d = sin_out.reshape(n_rows_rope, full_dim)

    BLOCK_SIZE_ROPE = triton.next_power_of_2(half_dim)
    grid_rope = (n_rows_rope,)

    rope_kernel[grid_rope](
        freqs_ptr=freqs_2d,
        cos_out_ptr=cos_2d,
        sin_out_ptr=sin_2d,
        half_dim=half_dim,
        n_rows=n_rows_rope,
        stride_freqs_row=freqs_2d.stride(0),
        stride_freqs_col=freqs_2d.stride(1),
        stride_cos_row=cos_2d.stride(0),
        stride_cos_col=cos_2d.stride(1),
        stride_sin_row=sin_2d.stride(0),
        stride_sin_col=sin_2d.stride(1),
        BLOCK_SIZE=BLOCK_SIZE_ROPE,
    )

    # === RMSNorm Computation ===
    hidden_dim = in_2.shape[-1]
    n_rows_norm = in_2.numel() // hidden_dim

    output = torch.empty_like(in_2, dtype=norm_dtype)

    input_2d = in_2.reshape(n_rows_norm, hidden_dim)
    output_2d = output.reshape(n_rows_norm, hidden_dim)

    BLOCK_SIZE_NORM = triton.next_power_of_2(hidden_dim)
    grid_norm = (n_rows_norm,)

    rmsnorm_kernel[grid_norm](
        input_ptr=input_2d,
        weight_ptr=in_0,
        output_ptr=output_2d,
        hidden_dim=hidden_dim,
        eps=eps,
        n_rows=n_rows_norm,
        stride_input_row=input_2d.stride(0),
        stride_input_col=input_2d.stride(1),
        stride_output_row=output_2d.stride(0),
        stride_output_col=output_2d.stride(1),
        BLOCK_SIZE=BLOCK_SIZE_NORM,
    )

    return (cos_out, output, sin_out)


# ===== Pattern for Variant A (bf16, eps=1e-06) =====

def pattern(in_0, in_1, in_2):
    tmp_1 = torch.cat((in_1, in_1), dim = -1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype = torch.bfloat16)
    tmp_7 = tmp_5.to(dtype = torch.bfloat16)
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim = True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_17 = in_0 * tmp_16
    return (tmp_6, tmp_17, tmp_7)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "route_bf16_eps1e6")


def replacement_func():
    return fused_rope_rmsnorm_dispatch