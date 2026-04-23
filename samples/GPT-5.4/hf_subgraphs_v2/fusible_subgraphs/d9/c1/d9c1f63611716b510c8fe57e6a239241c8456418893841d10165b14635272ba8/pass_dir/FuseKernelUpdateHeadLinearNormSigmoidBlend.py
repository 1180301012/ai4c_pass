import torch
import triton
import triton.language as tl


# Pattern matching function
# Must mirror model.py exactly in op sequence / argument style.
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
    linear = torch.nn.functional.linear(in_8, in_7, in_6)
    tmp_9 = torch.nn.functional.layer_norm(linear, (256,), in_3, in_2, 1e-05)
    tmp_10 = in_9.sigmoid()
    tmp_11 = tmp_9.sigmoid()
    tmp_12 = torch.nn.functional.layer_norm(in_11, (256,), in_5, in_4, 1e-05)
    tmp_13 = torch.nn.functional.layer_norm(in_10, (256,), in_1, in_0, 1e-05)
    tmp_14 = tmp_12.unsqueeze(-2)
    tmp_15 = tmp_11 * tmp_14
    tmp_16 = tmp_10 * tmp_13
    tmp_17 = tmp_15 + tmp_16
    return tmp_17


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11)


@triton.jit
def _linear_ln_sigmoid_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    ln_w_ptr,
    ln_b_ptr,
    out_ptr,
    n_rows,
    N_COLS: tl.constexpr,
    K_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return

    row_base_x = row * K_DIM
    row_base_out = row * N_COLS

    sum_y = tl.zeros([1], dtype=tl.float32)
    sumsq_y = tl.zeros([1], dtype=tl.float32)

    for n_start in range(0, N_COLS, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_COLS
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        for k_start in range(0, K_DIM, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            x = tl.load(x_ptr + row_base_x + offs_k, mask=offs_k < K_DIM, other=0).to(tl.float32)
            w = tl.load(
                w_ptr + offs_n[:, None] * K_DIM + offs_k[None, :],
                mask=mask_n[:, None] & (offs_k[None, :] < K_DIM),
                other=0,
            ).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)

        b = tl.load(b_ptr + offs_n, mask=mask_n, other=0).to(tl.float32)
        y = acc + b
        sum_y += tl.sum(y, axis=0)
        sumsq_y += tl.sum(y * y, axis=0)

    mean = sum_y / N_COLS
    var = sumsq_y / N_COLS - mean * mean
    var = tl.maximum(var, 0)
    inv_std = tl.rsqrt(var + 1e-5)

    for n_start in range(0, N_COLS, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_COLS
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        for k_start in range(0, K_DIM, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            x = tl.load(x_ptr + row_base_x + offs_k, mask=offs_k < K_DIM, other=0).to(tl.float32)
            w = tl.load(
                w_ptr + offs_n[:, None] * K_DIM + offs_k[None, :],
                mask=mask_n[:, None] & (offs_k[None, :] < K_DIM),
                other=0,
            ).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)

        b = tl.load(b_ptr + offs_n, mask=mask_n, other=0).to(tl.float32)
        gamma = tl.load(ln_w_ptr + offs_n, mask=mask_n, other=1).to(tl.float32)
        beta = tl.load(ln_b_ptr + offs_n, mask=mask_n, other=0).to(tl.float32)
        y = acc + b
        normed = (y - mean) * inv_std
        out = tl.sigmoid(normed * gamma + beta)
        tl.store(out_ptr + row_base_out + offs_n, out, mask=mask_n)


@triton.jit
def _tail_fused_kernel(
    gate_ptr,
    input_gate_ptr,
    input_out_ptr,
    param_out_ptr,
    in_out_ln_w_ptr,
    in_out_ln_b_ptr,
    param_ln_w_ptr,
    param_ln_b_ptr,
    out_ptr,
    n_rows,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return

    row_base = row * N_COLS

    sum_param = tl.zeros([1], dtype=tl.float32)
    sumsq_param = tl.zeros([1], dtype=tl.float32)
    sum_in = tl.zeros([1], dtype=tl.float32)
    sumsq_in = tl.zeros([1], dtype=tl.float32)

    for start in range(0, N_COLS, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N_COLS
        p = tl.load(param_out_ptr + row_base + offs, mask=mask, other=0).to(tl.float32)
        x = tl.load(input_out_ptr + row_base + offs, mask=mask, other=0).to(tl.float32)
        sum_param += tl.sum(p, axis=0)
        sumsq_param += tl.sum(p * p, axis=0)
        sum_in += tl.sum(x, axis=0)
        sumsq_in += tl.sum(x * x, axis=0)

    mean_param = sum_param / N_COLS
    var_param = sumsq_param / N_COLS - mean_param * mean_param
    var_param = tl.maximum(var_param, 0)
    inv_param = tl.rsqrt(var_param + 1e-5)

    mean_in = sum_in / N_COLS
    var_in = sumsq_in / N_COLS - mean_in * mean_in
    var_in = tl.maximum(var_in, 0)
    inv_in = tl.rsqrt(var_in + 1e-5)

    for start in range(0, N_COLS, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N_COLS

        gate = tl.load(gate_ptr + row_base + offs, mask=mask, other=0).to(tl.float32)
        inp_gate = tl.load(input_gate_ptr + row_base + offs, mask=mask, other=0).to(tl.float32)
        inp = tl.load(input_out_ptr + row_base + offs, mask=mask, other=0).to(tl.float32)
        param = tl.load(param_out_ptr + row_base + offs, mask=mask, other=0).to(tl.float32)

        in_w = tl.load(in_out_ln_w_ptr + offs, mask=mask, other=1).to(tl.float32)
        in_b = tl.load(in_out_ln_b_ptr + offs, mask=mask, other=0).to(tl.float32)
        param_w = tl.load(param_ln_w_ptr + offs, mask=mask, other=1).to(tl.float32)
        param_b = tl.load(param_ln_b_ptr + offs, mask=mask, other=0).to(tl.float32)

        inp_ln = ((inp - mean_in) * inv_in) * in_w + in_b
        param_ln = ((param - mean_param) * inv_param) * param_w + param_b
        out = gate * param_ln + tl.sigmoid(inp_gate) * inp_ln
        tl.store(out_ptr + row_base + offs, out, mask=mask)


@torch.fx.wrap
def fused_kernel_update_head_linear_norm_sigmoid_blend(
    in_0,
    in_1,
    in_2,
    in_3,
    in_4,
    in_5,
    in_6,
    in_7,
    in_8,
    in_9,
    in_10,
    in_11,
):
    n_cols = in_10.shape[-1]
    n_rows = in_10.numel() // n_cols

    gate = torch.empty_like(in_10)
    out = torch.empty_like(in_10)

    grid = (n_rows,)

    _linear_ln_sigmoid_kernel[grid](
        in_8,
        in_7,
        in_6,
        in_3,
        in_2,
        gate,
        n_rows,
        N_COLS=256,
        K_DIM=256,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=8,
        num_stages=2,
    )

    _tail_fused_kernel[grid](
        gate,
        in_9,
        in_10,
        in_11,
        in_1,
        in_0,
        in_5,
        in_4,
        out,
        n_rows,
        N_COLS=256,
        BLOCK=128,
        num_warps=4,
        num_stages=2,
    )

    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_kernel_update_head_linear_norm_sigmoid_blend