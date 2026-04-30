import torch
import triton
import triton.language as tl


def pattern(e0, e1, e2, e3, e4, e5, e6, e7, e8, weight, bias):
    t1 = e0 + e1
    t2 = t1 + e2
    t3 = t2 + e3
    t4 = t3 + e4
    t5 = t4 + e5
    t6 = t5 + e6
    t7 = t6 + e7
    t8 = t7 + e8
    ln = torch.nn.functional.layer_norm(t8, (768,), weight, bias, 1e-12)
    result = torch.nn.functional.dropout(ln, 0.1, False, False)
    return result


def replacement_args(e0, e1, e2, e3, e4, e5, e6, e7, e8, weight, bias):
    return (e0, e1, e2, e3, e4, e5, e6, e7, e8, weight, bias)


@triton.jit
def fused_sum_layernorm_kernel(
    e0_ptr, e1_ptr, e2_ptr, e3_ptr, e4_ptr, e5_ptr, e6_ptr, e7_ptr, e8_ptr,
    weight_ptr, bias_ptr, output_ptr,
    n_rows,
    HIDDEN_DIM: tl.constexpr,
    eps,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, HIDDEN_DIM)
    row_start = row_idx * HIDDEN_DIM

    # Accumulate embeddings in float32 for numerical stability
    acc = tl.zeros([HIDDEN_DIM], tl.float32)
    acc += tl.load(e0_ptr + row_start + col_offsets).to(tl.float32)
    acc += tl.load(e1_ptr + row_start + col_offsets).to(tl.float32)
    acc += tl.load(e2_ptr + row_start + col_offsets).to(tl.float32)
    acc += tl.load(e3_ptr + row_start + col_offsets).to(tl.float32)
    acc += tl.load(e4_ptr + row_start + col_offsets).to(tl.float32)
    acc += tl.load(e5_ptr + row_start + col_offsets).to(tl.float32)
    acc += tl.load(e6_ptr + row_start + col_offsets).to(tl.float32)
    acc += tl.load(e7_ptr + row_start + col_offsets).to(tl.float32)
    acc += tl.load(e8_ptr + row_start + col_offsets).to(tl.float32)

    # Compute mean
    mean = tl.sum(acc) / HIDDEN_DIM

    # Compute variance
    diff = acc - mean
    variance = tl.sum(diff * diff) / HIDDEN_DIM

    # Normalize
    rstd = 1.0 / tl.sqrt(variance + eps)
    normalized = diff * rstd

    # Apply weight and bias
    weight = tl.load(weight_ptr + col_offsets).to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets).to(tl.float32)
    output = normalized * weight + bias

    # Store result
    tl.store(output_ptr + row_start + col_offsets, output)


@torch.fx.wrap
def fused_sum_layernorm(e0, e1, e2, e3, e4, e5, e6, e7, e8, weight, bias):
    HIDDEN_DIM = 768
    n_rows = e0.shape[0] * e0.shape[1]

    # Create output with same dtype and shape as input embeddings
    output = torch.empty_like(e0)

    # Flatten to 2D for kernel processing
    e0_2d = e0.reshape(-1, HIDDEN_DIM)
    e1_2d = e1.reshape(-1, HIDDEN_DIM)
    e2_2d = e2.reshape(-1, HIDDEN_DIM)
    e3_2d = e3.reshape(-1, HIDDEN_DIM)
    e4_2d = e4.reshape(-1, HIDDEN_DIM)
    e5_2d = e5.reshape(-1, HIDDEN_DIM)
    e6_2d = e6.reshape(-1, HIDDEN_DIM)
    e7_2d = e7.reshape(-1, HIDDEN_DIM)
    e8_2d = e8.reshape(-1, HIDDEN_DIM)
    output_2d = output.reshape(-1, HIDDEN_DIM)

    grid = (n_rows,)

    fused_sum_layernorm_kernel[grid](
        e0_2d, e1_2d, e2_2d, e3_2d, e4_2d, e5_2d, e6_2d, e7_2d, e8_2d,
        weight, bias, output_2d,
        n_rows=n_rows,
        HIDDEN_DIM=HIDDEN_DIM,
        eps=1e-12,
        num_warps=4,
    )

    return output


def replacement_func():
    return fused_sum_layernorm