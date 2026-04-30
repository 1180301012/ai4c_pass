import torch
import triton
import triton.language as tl


# ===== Pattern Matching (RMSNorm: eps=1e-06, output=bfloat16) =====

def pattern(in_0, in_2):
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_17 = in_0 * tmp_16
    return tmp_17


def replacement_args(in_0, in_2):
    return (in_0, in_2)


# ===== Triton Kernel =====

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

    # Store (Triton handles dtype casting)
    tl.store(output_ptr + output_base + col_offsets, result, mask=mask)


# ===== Kernel Wrapper =====

@torch.fx.wrap
def fused_rmsnorm_bf16(in_0, in_2):
    eps = 1e-06
    output_dtype = torch.bfloat16

    # in_2 shape: [B, S, H]
    B = in_2.shape[0]
    S_ln = in_2.shape[1]
    H = in_2.shape[2]

    ln_out = torch.empty((B, S_ln, H), dtype=output_dtype, device=in_2.device)

    BLOCK_SIZE = triton.next_power_of_2(H)

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
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return ln_out


def replacement_func():
    return fused_rmsnorm_bf16