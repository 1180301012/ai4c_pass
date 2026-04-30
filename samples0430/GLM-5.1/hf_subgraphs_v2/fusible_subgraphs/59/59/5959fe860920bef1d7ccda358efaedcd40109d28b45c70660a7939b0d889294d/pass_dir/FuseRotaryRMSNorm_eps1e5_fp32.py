import torch
import triton
import triton.language as tl


# ===== Pattern Matching (TinyLlama variant: eps=1e-05, output=float32) =====

def pattern(in_0, in_1, in_2):
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.float32)
    tmp_7 = tmp_5.to(dtype=torch.float32)
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-05
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.float32)
    tmp_17 = in_0 * tmp_16
    return (tmp_6, tmp_17, tmp_7)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ===== Triton Kernels =====

@triton.jit
def rotary_emb_kernel(
    freqs_ptr,
    cos_out_ptr,
    sin_out_ptr,
    n_rows,
    D,
    freqs_stride_1,
    cos_stride_1,
    sin_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    s = tl.program_id(0)
    if s >= n_rows:
        return

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D

    # Load freqs values and cast to float32
    freqs_val = tl.load(freqs_ptr + s * freqs_stride_1 + col_offsets, mask=mask, other=0.0)
    freqs_val = freqs_val.to(tl.float32)

    # Compute cos and sin in float32
    cos_val = tl.cos(freqs_val)
    sin_val = tl.sin(freqs_val)

    # Store to first half [0, D)
    tl.store(cos_out_ptr + s * cos_stride_1 + col_offsets, cos_val, mask=mask)
    tl.store(sin_out_ptr + s * sin_stride_1 + col_offsets, sin_val, mask=mask)

    # Store to second half [D, 2*D)
    tl.store(cos_out_ptr + s * cos_stride_1 + col_offsets + D, cos_val, mask=mask)
    tl.store(sin_out_ptr + s * sin_stride_1 + col_offsets + D, sin_val, mask=mask)


@triton.jit
def rmsnorm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    H,
    eps,
    stride_b,
    stride_s,
    out_stride_b,
    out_stride_s,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < H

    # Compute base offsets for this row
    input_base = b * stride_b + s * stride_s
    output_base = b * out_stride_b + s * out_stride_s

    # Load input row as float32
    input_row = tl.load(input_ptr + input_base + col_offsets, mask=mask, other=0.0)
    input_row = input_row.to(tl.float32)

    # Compute mean of squares
    sq_row = input_row * input_row
    mean_sq = tl.sum(sq_row, axis=0) / H

    # Compute rsqrt(mean_sq + eps)
    norm_factor = tl.rsqrt(mean_sq + eps)

    # Normalize
    normalized = input_row * norm_factor

    # Load weight as float32
    weight_row = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    weight_row = weight_row.to(tl.float32)

    # Multiply by weight
    result = normalized * weight_row

    # Store
    tl.store(output_ptr + output_base + col_offsets, result, mask=mask)


# ===== Kernel Wrapper =====

@torch.fx.wrap
def fused_rotary_rmsnorm_fp32(in_0, in_1, in_2):
    eps = 1e-05
    output_dtype = torch.float32

    # === Rotary Embedding ===
    S_freqs = in_1.shape[1]
    D = in_1.shape[2]
    out_last_dim = 2 * D

    cos_out = torch.empty((1, S_freqs, out_last_dim), dtype=output_dtype, device=in_1.device)
    sin_out = torch.empty((1, S_freqs, out_last_dim), dtype=output_dtype, device=in_1.device)

    BLOCK_SIZE_ROTARY = triton.next_power_of_2(D)

    rotary_emb_kernel[(S_freqs,)](
        freqs_ptr=in_1,
        cos_out_ptr=cos_out,
        sin_out_ptr=sin_out,
        n_rows=S_freqs,
        D=D,
        freqs_stride_1=in_1.stride(1),
        cos_stride_1=cos_out.stride(1),
        sin_stride_1=sin_out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE_ROTARY,
    )

    # === RMSNorm ===
    B = in_2.shape[0]
    S_ln = in_2.shape[1]
    H = in_2.shape[2]

    ln_out = torch.empty((B, S_ln, H), dtype=output_dtype, device=in_2.device)

    BLOCK_SIZE_RMSNORM = triton.next_power_of_2(H)

    rmsnorm_kernel[(B, S_ln)](
        input_ptr=in_2,
        weight_ptr=in_0,
        output_ptr=ln_out,
        H=H,
        eps=eps,
        stride_b=in_2.stride(0),
        stride_s=in_2.stride(1),
        out_stride_b=ln_out.stride(0),
        out_stride_s=ln_out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE_RMSNORM,
    )

    return (cos_out, ln_out, sin_out)


def replacement_func():
    return fused_rotary_rmsnorm_fp32