import torch
import triton
import triton.language as tl


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
def fused_whole_graph_kernel(
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


@triton.jit
def tail_gated_add_kernel(
    tmp11_ptr,
    tmp12_ptr,
    tmp10_ptr,
    tmp13_ptr,
    out_ptr,
    stride_tmp11_0,
    stride_tmp11_2,
    stride_tmp12_0,
    stride_tmp12_1,
    stride_tmp10_0,
    stride_tmp10_2,
    stride_tmp13_0,
    stride_tmp13_2,
    stride_out_0,
    stride_out_2,
    HIDDEN: tl.constexpr,
):
    row = tl.program_id(0)
    offs_n = tl.arange(0, HIDDEN)
    a = tl.load(tmp11_ptr + row * stride_tmp11_0 + offs_n * stride_tmp11_2).to(tl.float32)
    b = tl.load(tmp12_ptr + row * stride_tmp12_0 + offs_n * stride_tmp12_1).to(tl.float32)
    c = tl.load(tmp10_ptr + row * stride_tmp10_0 + offs_n * stride_tmp10_2).to(tl.float32)
    d = tl.load(tmp13_ptr + row * stride_tmp13_0 + offs_n * stride_tmp13_2).to(tl.float32)
    out = a * b + c * d
    tl.store(out_ptr + row * stride_out_0 + offs_n * stride_out_2, out)


@torch.fx.wrap
def shared_dispatch(*args):
    route = args[-1]

    if route == 'whole_graph':
        in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, _ = args
        out = torch.empty_like(in_9)
        fused_whole_graph_kernel[(in_11.shape[0],)](
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
            BLOCK_K=64 if in_9.dtype != torch.float32 else 32,
            EPS=1e-5,
            num_warps=8 if in_9.dtype != torch.float32 else 4,
            num_stages=2,
        )
        return out

    if route == 'tail_gated_add':
        tmp11, tmp12, tmp10, tmp13, _ = args
        out = torch.empty_like(tmp11)
        tail_gated_add_kernel[(tmp11.shape[0],)](
            tmp11,
            tmp12,
            tmp10,
            tmp13,
            out,
            tmp11.stride(0),
            tmp11.stride(2),
            tmp12.stride(0),
            tmp12.stride(1),
            tmp10.stride(0),
            tmp10.stride(2),
            tmp13.stride(0),
            tmp13.stride(2),
            out.stride(0),
            out.stride(2),
            HIDDEN=256,
            num_warps=4,
            num_stages=2,
        )
        return out

    raise RuntimeError(f'Unknown route: {route}')


def replacement_func():
    return shared_dispatch