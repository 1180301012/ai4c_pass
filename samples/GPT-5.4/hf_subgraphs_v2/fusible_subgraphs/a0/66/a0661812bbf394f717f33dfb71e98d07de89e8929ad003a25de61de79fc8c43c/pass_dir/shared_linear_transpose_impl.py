import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.utils._mode_utils import no_dispatch

from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    ],
    key=["M"],
)
@triton.jit
def _tiny_linear_bias_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xb,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ob,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    x_batch_ptr = x_ptr + pid_b * stride_xb
    out_batch_ptr = out_ptr + pid_b * stride_ob

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_batch_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w))

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    tl.store(
        out_batch_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _should_use_tiny_kernel(bias, weight, x):
    with no_dispatch():
        raw_bias = unwrap_tensor(bias)
        raw_weight = unwrap_tensor(weight)
        raw_x = unwrap_tensor(x)
        return (
            raw_x.is_cuda
            and raw_x.dim() == 3
            and raw_weight.dim() == 2
            and raw_bias is not None
            and raw_bias.dim() == 1
            and raw_x.size(0) == 1
            and raw_x.size(-1) == 32
            and raw_weight.size(0) <= 32
            and raw_weight.size(1) == raw_x.size(-1)
            and raw_bias.numel() == raw_weight.size(0)
            and raw_x.is_contiguous()
            and raw_weight.is_contiguous()
            and raw_bias.is_contiguous()
            and raw_x.dtype in (torch.float16, torch.bfloat16)
            and raw_weight.dtype == raw_x.dtype
            and raw_bias.dtype == raw_x.dtype
        )


@torch.fx.wrap
def shared_linear_dropout_transpose_dispatch(bias, weight, x, route):
    with no_dispatch():
        raw_bias = unwrap_tensor(bias)
        raw_weight = unwrap_tensor(weight)
        raw_x = unwrap_tensor(x)

        if _should_use_tiny_kernel(raw_bias, raw_weight, raw_x):
            batch = raw_x.size(0)
            m_dim = raw_x.size(1)
            k_dim = raw_x.size(2)
            n_dim = raw_weight.size(0)
            out = torch.empty((batch, m_dim, n_dim), device=raw_x.device, dtype=raw_x.dtype)

            grid = lambda META: (triton.cdiv(m_dim, META["BLOCK_M"]), batch)
            _tiny_linear_bias_kernel[grid](
                raw_x,
                raw_weight,
                raw_bias,
                out,
                m_dim,
                n_dim,
                k_dim,
                raw_x.stride(0),
                raw_x.stride(1),
                raw_x.stride(2),
                raw_weight.stride(0),
                raw_weight.stride(1),
                out.stride(0),
                out.stride(1),
                out.stride(2),
            )
        else:
            out = F.linear(raw_x, raw_weight, raw_bias)

        out_t = out.transpose(1, 2)

        if route == "orig_transposed":
            return out, out_t
        if route == "transposed_orig":
            return out_t, out
        raise RuntimeError(f"Unknown route: {route}")


@torch.fx.wrap
def shared_linear_identity_dropout_dispatch(bias, weight, x):
    with no_dispatch():
        raw_bias = unwrap_tensor(bias)
        raw_weight = unwrap_tensor(weight)
        raw_x = unwrap_tensor(x)

        if _should_use_tiny_kernel(raw_bias, raw_weight, raw_x):
            batch = raw_x.size(0)
            m_dim = raw_x.size(1)
            k_dim = raw_x.size(2)
            n_dim = raw_weight.size(0)
            out = torch.empty((batch, m_dim, n_dim), device=raw_x.device, dtype=raw_x.dtype)

            grid = lambda META: (triton.cdiv(m_dim, META["BLOCK_M"]), batch)
            _tiny_linear_bias_kernel[grid](
                raw_x,
                raw_weight,
                raw_bias,
                out,
                m_dim,
                n_dim,
                k_dim,
                raw_x.stride(0),
                raw_x.stride(1),
                raw_x.stride(2),
                raw_weight.stride(0),
                raw_weight.stride(1),
                out.stride(0),
                out.stride(1),
                out.stride(2),
            )
            return out

        return F.linear(raw_x, raw_weight, raw_bias)