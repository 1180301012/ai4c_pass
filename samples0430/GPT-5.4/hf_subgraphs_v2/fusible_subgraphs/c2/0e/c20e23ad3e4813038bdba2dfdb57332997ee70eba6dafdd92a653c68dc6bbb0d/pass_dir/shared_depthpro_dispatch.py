import torch
import triton
import triton.language as tl

ROUTE_IDENTITY = "identity"
ROUTE_ADD = "add"
ROUTE_CONV_RELU = "conv_relu"
ROUTE_CONV_RELU_ADD = "conv_relu_add"


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=2),
    ],
    key=["M", "K_OUT"],
)
@triton.jit
def _depthpro_conv_relu_add_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    r_ptr,
    o_ptr,
    M,
    K_OUT,
    x_s_c,
    x_s_h,
    x_s_w,
    w_s_o,
    w_s_i,
    w_s_h,
    w_s_w,
    r_s_c,
    r_s_h,
    r_s_w,
    o_s_c,
    o_s_h,
    o_s_w,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    C_IN: tl.constexpr,
    W_OUT: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(K_OUT, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    out_h = offs_m // W_OUT
    out_w = offs_m % W_OUT
    base_h = out_h * STRIDE_H - PAD_H
    base_w = out_w * STRIDE_W - PAD_W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_total = C_IN * K_H * K_W

    for k_start in range(0, k_total, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        c = offs_k // (K_H * K_W)
        khkw = offs_k % (K_H * K_W)
        kh = khkw // K_W
        kw = khkw % K_W

        in_h = base_h[:, None] + kh[None, :]
        in_w = base_w[:, None] + kw[None, :]

        x_ptrs = x_ptr + c[None, :] * x_s_c + in_h * x_s_h + in_w * x_s_w
        x_mask = (
            (offs_m[:, None] < M)
            & (c[None, :] < C_IN)
            & (in_h >= 0)
            & (in_h < H_IN)
            & (in_w >= 0)
            & (in_w < W_IN)
        )
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = w_ptr + offs_n[None, :] * w_s_o + c[:, None] * w_s_i + kh[:, None] * w_s_h + kw[:, None] * w_s_w
        w_mask = (offs_n[None, :] < K_OUT) & (c[:, None] < C_IN)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x, w, acc)

    bias = tl.load(b_ptr + offs_n, mask=offs_n < K_OUT, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)
    conv_out = acc.to(tl.float16)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K_OUT)
    r_ptrs = r_ptr + offs_n[None, :] * r_s_c + out_h[:, None] * r_s_h + out_w[:, None] * r_s_w
    residual = tl.load(r_ptrs, mask=out_mask, other=0.0)
    out = conv_out + residual

    o_ptrs = o_ptr + offs_n[None, :] * o_s_c + out_h[:, None] * o_s_h + out_w[:, None] * o_s_w
    tl.store(o_ptrs, out, mask=out_mask)


@torch.fx.wrap
def depthpro_shared_dispatch(*args):
    route = args[-1]
    if route == ROUTE_IDENTITY:
        return args[0]
    if route == ROUTE_ADD:
        x, y = args[0], args[1]
        out = torch.empty_like(x)
        n = x.numel()
        grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)
        _add_kernel[grid](x, y, out, n)
        return out
    if route == ROUTE_CONV_RELU:
        return torch.nn.functional.relu(torch.conv2d(args[3], args[1], args[0], (2, 2), (1, 1), (1, 1), 1), inplace=True)
    if route == ROUTE_CONV_RELU_ADD:
        bias, weight, residual, x = args[0], args[1], args[2], args[3]
        out = torch.empty_like(residual)
        m = residual.shape[2] * residual.shape[3]
        k_out = residual.shape[1]
        grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(k_out, META["BLOCK_N"]),)
        _depthpro_conv_relu_add_kernel[grid](
            x,
            weight,
            bias,
            residual,
            out,
            m,
            k_out,
            x.stride(1),
            x.stride(2),
            x.stride(3),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            weight.stride(3),
            residual.stride(1),
            residual.stride(2),
            residual.stride(3),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            H_IN=x.shape[2],
            W_IN=x.shape[3],
            C_IN=x.shape[1],
            W_OUT=residual.shape[3],
            K_H=weight.shape[2],
            K_W=weight.shape[3],
            STRIDE_H=2,
            STRIDE_W=2,
            PAD_H=1,
            PAD_W=1,
        )
        return out
    return args[0]