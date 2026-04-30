import torch
import triton
import triton.language as tl


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
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _layer_norm_affine(vec, weight_ptr, bias_ptr, offs_n, HIDDEN: tl.constexpr, EPS: tl.constexpr):
    scale = 1.0 / HIDDEN
    mean = tl.sum(vec, axis=0) * scale
    diff = vec - mean
    var = tl.sum(diff * diff, axis=0) * scale
    inv_std = tl.rsqrt(var + EPS)
    gamma = tl.load(weight_ptr + offs_n).to(tl.float32)
    beta = tl.load(bias_ptr + offs_n).to(tl.float32)
    return diff * inv_std * gamma + beta


@triton.jit
def _layer_norm_affine_block(mat, weight_ptr, bias_ptr, offs_n, HIDDEN: tl.constexpr, EPS: tl.constexpr):
    scale = 1.0 / HIDDEN
    mean = tl.sum(mat, axis=1) * scale
    diff = mat - mean[:, None]
    var = tl.sum(diff * diff, axis=1) * scale
    inv_std = tl.rsqrt(var + EPS)
    gamma = tl.load(weight_ptr + offs_n).to(tl.float32)
    beta = tl.load(bias_ptr + offs_n).to(tl.float32)
    return diff * inv_std[:, None] * gamma[None, :] + beta[None, :]


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M'],
)
@triton.jit
def fused_kernel_update_half_kernel(
    bias0_ptr,
    weight1_ptr,
    bias2_ptr,
    weight3_ptr,
    bias4_ptr,
    weight5_ptr,
    bias6_ptr,
    weight7_ptr,
    in8_ptr,
    in9_ptr,
    in10_ptr,
    in11_ptr,
    out_ptr,
    M,
    stride_w7_0,
    stride_w7_1,
    stride_in8_0,
    stride_in8_2,
    stride_in9_0,
    stride_in9_2,
    stride_in10_0,
    stride_in10_2,
    stride_in11_0,
    stride_in11_1,
    stride_out_0,
    stride_out_2,
    HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, HIDDEN)

    acc = tl.zeros((BLOCK_M, HIDDEN), dtype=tl.float32)
    for k_start in range(0, HIDDEN, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(
            in8_ptr + offs_m[:, None] * stride_in8_0 + offs_k[None, :] * stride_in8_2,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < HIDDEN),
            other=0.0,
        )
        w = tl.load(
            weight7_ptr + offs_n[:, None] * stride_w7_0 + offs_k[None, :] * stride_w7_1,
            mask=(offs_n[:, None] < HIDDEN) & (offs_k[None, :] < HIDDEN),
            other=0.0,
        )
        acc += tl.dot(a, tl.trans(w))

    bias6 = tl.load(bias6_ptr + offs_n).to(tl.float32)
    linear = acc + bias6[None, :]
    gate_update = _layer_norm_affine_block(linear, weight3_ptr, bias2_ptr, offs_n, HIDDEN, EPS)
    gate_update_sigmoid = _sigmoid(gate_update)

    input_gate = tl.load(
        in9_ptr + offs_m[:, None] * stride_in9_0 + offs_n[None, :] * stride_in9_2,
        mask=offs_m[:, None] < M,
        other=0.0,
    ).to(tl.float32)
    input_gate_sigmoid = _sigmoid(input_gate)

    input_out = tl.load(
        in10_ptr + offs_m[:, None] * stride_in10_0 + offs_n[None, :] * stride_in10_2,
        mask=offs_m[:, None] < M,
        other=0.0,
    ).to(tl.float32)
    input_out_norm = _layer_norm_affine_block(input_out, weight1_ptr, bias0_ptr, offs_n, HIDDEN, EPS)

    param_out = tl.load(
        in11_ptr + offs_m[:, None] * stride_in11_0 + offs_n[None, :] * stride_in11_1,
        mask=offs_m[:, None] < M,
        other=0.0,
    ).to(tl.float32)
    param_out_norm = _layer_norm_affine_block(param_out, weight5_ptr, bias4_ptr, offs_n, HIDDEN, EPS)

    out = gate_update_sigmoid * param_out_norm + input_gate_sigmoid * input_out_norm
    tl.store(
        out_ptr + offs_m[:, None] * stride_out_0 + offs_n[None, :] * stride_out_2,
        out,
        mask=offs_m[:, None] < M,
    )


@triton.jit
def fused_kernel_update_float_kernel(
    bias0_ptr,
    weight1_ptr,
    bias2_ptr,
    weight3_ptr,
    bias4_ptr,
    weight5_ptr,
    bias6_ptr,
    weight7_ptr,
    in8_ptr,
    in9_ptr,
    in10_ptr,
    in11_ptr,
    out_ptr,
    stride_w7_0,
    stride_w7_1,
    stride_in8_0,
    stride_in8_2,
    stride_in9_0,
    stride_in9_2,
    stride_in10_0,
    stride_in10_2,
    stride_in11_0,
    stride_in11_1,
    stride_out_0,
    stride_out_2,
    HIDDEN: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    offs_n = tl.arange(0, HIDDEN)

    acc = tl.zeros((HIDDEN,), dtype=tl.float32)
    for k_start in range(0, HIDDEN, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x = tl.load(in8_ptr + row * stride_in8_0 + offs_k * stride_in8_2, mask=offs_k < HIDDEN, other=0.0).to(tl.float32)
        w = tl.load(
            weight7_ptr + offs_n[:, None] * stride_w7_0 + offs_k[None, :] * stride_w7_1,
            mask=(offs_n[:, None] < HIDDEN) & (offs_k[None, :] < HIDDEN),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    linear = acc + tl.load(bias6_ptr + offs_n).to(tl.float32)
    gate_update = _layer_norm_affine(linear, weight3_ptr, bias2_ptr, offs_n, HIDDEN, EPS)
    gate_update_sigmoid = _sigmoid(gate_update)

    input_gate = tl.load(in9_ptr + row * stride_in9_0 + offs_n * stride_in9_2).to(tl.float32)
    input_gate_sigmoid = _sigmoid(input_gate)

    input_out = tl.load(in10_ptr + row * stride_in10_0 + offs_n * stride_in10_2).to(tl.float32)
    input_out_norm = _layer_norm_affine(input_out, weight1_ptr, bias0_ptr, offs_n, HIDDEN, EPS)

    param_out = tl.load(in11_ptr + row * stride_in11_0 + offs_n * stride_in11_1).to(tl.float32)
    param_out_norm = _layer_norm_affine(param_out, weight5_ptr, bias4_ptr, offs_n, HIDDEN, EPS)

    out = gate_update_sigmoid * param_out_norm + input_gate_sigmoid * input_out_norm
    tl.store(out_ptr + row * stride_out_0 + offs_n * stride_out_2, out)


@torch.fx.wrap
def fused_kernel_update(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
    out = torch.empty_like(in_9)
    hidden = in_7.shape[1]
    if hidden != 256:
        raise RuntimeError(f"Expected hidden size 256, got {hidden}")

    if in_9.dtype == torch.float32:
        fused_kernel_update_float_kernel[(in_11.shape[0],)](
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
            out,
            in_7.stride(0),
            in_7.stride(1),
            in_8.stride(0),
            in_8.stride(2),
            in_9.stride(0),
            in_9.stride(2),
            in_10.stride(0),
            in_10.stride(2),
            in_11.stride(0),
            in_11.stride(1),
            out.stride(0),
            out.stride(2),
            HIDDEN=256,
            BLOCK_K=32,
            EPS=1e-5,
            num_warps=4,
            num_stages=2,
        )
    else:
        grid = lambda meta: (triton.cdiv(in_11.shape[0], meta['BLOCK_M']),)
        fused_kernel_update_half_kernel[grid](
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
            out,
            in_11.shape[0],
            in_7.stride(0),
            in_7.stride(1),
            in_8.stride(0),
            in_8.stride(2),
            in_9.stride(0),
            in_9.stride(2),
            in_10.stride(0),
            in_10.stride(2),
            in_11.stride(0),
            in_11.stride(1),
            out.stride(0),
            out.stride(2),
            HIDDEN=256,
            EPS=1e-5,
        )
    return out


def replacement_func():
    return fused_kernel_update