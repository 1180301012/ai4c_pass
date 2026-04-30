import torch
import triton
import triton.language as tl


@triton.jit
def _identity_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, x, mask=mask)


_TINY_CONFIGS = [
    triton.Config({"BLOCK_M": 64}, num_stages=2, num_warps=2),
    triton.Config({"BLOCK_M": 128}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 256}, num_stages=2, num_warps=4),
]


@triton.autotune(configs=_TINY_CONFIGS, key=["T"])
@triton.jit
def _tiny_linear_dual_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    out_t_ptr,
    T,
    stride_xt,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ot,
    stride_on,
    stride_tn,
    stride_tt,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, 32)
    offs_n = tl.arange(0, 16)

    x_ptrs = x_ptr + offs_t[:, None] * stride_xt + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    x = tl.load(x_ptrs, mask=offs_t[:, None] < T, other=0.0)
    w = tl.load(w_ptrs)
    acc = tl.dot(x, w)
    bias = tl.load(bias_ptr + offs_n).to(tl.float32)
    acc += bias[None, :]

    out_ptrs = out_ptr + offs_t[:, None] * stride_ot + offs_n[None, :] * stride_on
    out_mask = offs_t[:, None] < T
    tl.store(out_ptrs, acc, mask=out_mask)

    out_t_ptrs = out_t_ptr + offs_n[:, None] * stride_tn + offs_t[None, :] * stride_tt
    out_t_mask = offs_t[None, :] < T
    tl.store(out_t_ptrs, tl.trans(acc), mask=out_t_mask)


def _tiny_swapped_impl(bias, weight, x):
    t = x.shape[1]
    out = torch.empty((1, t, 16), device=x.device, dtype=x.dtype)
    out_t = torch.empty((1, 16, t), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(t, META["BLOCK_M"]),)
    _tiny_linear_dual_kernel[grid](
        x,
        weight,
        bias,
        out,
        out_t,
        t,
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        out.stride(1),
        out.stride(2),
        out_t.stride(1),
        out_t.stride(2),
    )
    return out_t, out


@torch.fx.wrap
def shared_replacement(*args):
    route = args[-1]
    if route == "dropout_identity":
        return args[0]
    if route == "tiny_swapped_p0_0":
        return _tiny_swapped_impl(args[0], args[1], args[2])
    raise RuntimeError(f"Unknown route: {route}")