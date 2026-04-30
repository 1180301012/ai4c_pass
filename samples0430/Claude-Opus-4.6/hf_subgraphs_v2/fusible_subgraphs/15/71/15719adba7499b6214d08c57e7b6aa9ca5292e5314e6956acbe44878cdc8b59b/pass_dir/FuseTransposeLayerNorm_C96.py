import torch
import triton
import triton.language as tl


def pattern(conv_out, normalized_shape, weight, bias):
    tmp_6 = conv_out.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, normalized_shape, weight, bias, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    return tmp_9


def replacement_args(conv_out, normalized_shape, weight, bias):
    return (conv_out, weight, bias)


@triton.jit
def fused_transpose_layernorm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    C: tl.constexpr,
    HW,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Each program handles BLOCK_N spatial positions
    pid = tl.program_id(0)
    row_start = pid * BLOCK_N

    c_offsets = tl.arange(0, BLOCK_C)  # [BLOCK_C]
    n_offsets = tl.arange(0, BLOCK_N)  # [BLOCK_N]
    c_mask = c_offsets < C

    rows = row_start + n_offsets  # [BLOCK_N]
    row_mask = rows < HW

    # Load weight and bias
    w = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

    # Load from NCHW layout with [BLOCK_C, BLOCK_N] layout for coalesced reads
    # address[c, n] = c * HW + (row_start + n)
    # Consecutive n values → consecutive memory addresses → coalesced!
    load_offsets = c_offsets[:, None] * HW + rows[None, :]  # [BLOCK_C, BLOCK_N]
    load_mask = c_mask[:, None] & row_mask[None, :]
    x = tl.load(input_ptr + load_offsets, mask=load_mask, other=0.0)
    x_f32 = x.to(tl.float32)  # [BLOCK_C, BLOCK_N]

    # Compute mean per spatial position: reduce over C dimension (axis=0)
    mean = tl.sum(x_f32, axis=0) / C  # [BLOCK_N]

    # Compute variance - mask to avoid padding contamination
    diff = x_f32 - mean[None, :]  # [BLOCK_C, BLOCK_N]
    diff = tl.where(c_mask[:, None], diff, 0.0)
    var = tl.sum(diff * diff, axis=0) / C  # [BLOCK_N]

    # Normalize
    inv_std = 1.0 / tl.sqrt(var + 1e-5)  # [BLOCK_N]
    x_norm = diff * inv_std[None, :]  # [BLOCK_C, BLOCK_N]

    # Apply affine transform
    out = x_norm * w[:, None] + b[:, None]  # [BLOCK_C, BLOCK_N]

    # Store in [B, HW, C] layout: output[row, c] = output_ptr[row * C + c]
    store_offsets = rows[None, :] * C + c_offsets[:, None]  # [BLOCK_C, BLOCK_N]
    store_mask = c_mask[:, None] & row_mask[None, :]
    tl.store(output_ptr + store_offsets, out.to(x.dtype), mask=store_mask)


@torch.fx.wrap
def fused_transpose_layernorm(conv_out, weight, bias):
    B = conv_out.shape[0]
    C = conv_out.shape[1]
    H = conv_out.shape[2]
    W = conv_out.shape[3]
    HW = H * W
    output = torch.empty(B, HW, C, dtype=conv_out.dtype, device=conv_out.device)

    BLOCK_C = triton.next_power_of_2(C)
    if HW <= 1024:
        BLOCK_N = 32
    else:
        BLOCK_N = 64
    num_programs = (HW + BLOCK_N - 1) // BLOCK_N
    fused_transpose_layernorm_kernel[(num_programs,)](
        conv_out, weight, bias, output,
        C=C, HW=HW, BLOCK_C=BLOCK_C, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )

    return output


def replacement_func():
    return fused_transpose_layernorm