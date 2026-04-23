import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=[],
)
@triton.jit
def _bigbird_linear_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, 768, BLOCK_K):
        x = tl.load(x_ptrs, mask=offs_m[:, None] < 17, other=0.0)
        w = tl.load(w_ptrs, mask=offs_n[:, None] < 3072, other=0.0)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    bias = tl.load(b_ptr + offs_n, mask=offs_n < 3072, other=0.0).to(tl.float32)
    acc += bias[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < 17) & (offs_n[None, :] < 3072)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=[],
)
@triton.jit
def _rect_linear_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, 128, BLOCK_K):
        x = tl.load(x_ptrs, mask=offs_m[:, None] < 128, other=0.0)
        w = tl.load(w_ptrs, mask=offs_n[:, None] < 128, other=0.0)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    bias = tl.load(b_ptr + offs_n, mask=offs_n < 128, other=0.0).to(tl.float32)
    acc += bias[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < 128) & (offs_n[None, :] < 128)
    tl.store(out_ptrs, acc, mask=out_mask)


@torch.fx.wrap
def shared_dispatch(*args):
    route = args[-1]

    if route == 'identity':
        return args[0]

    if route != 'linear':
        raise RuntimeError('Unknown route in shared_dispatch')

    b = args[0]
    w = args[1]
    x = args[2]
    x_shape = x.shape

    if len(x_shape) == 3 and x_shape[0] == 1 and x_shape[1] == 17 and x_shape[2] == 768 and w.shape[0] == 3072 and w.shape[1] == 768:
        out = torch.empty((1, 17, 3072), device=x.device, dtype=x.dtype)
        stride_x = x.stride()
        stride_w = w.stride()
        stride_o = out.stride()
        grid = lambda META: (triton.cdiv(3072, META['BLOCK_N']), triton.cdiv(17, META['BLOCK_M']))
        _bigbird_linear_kernel[grid](
            x,
            w,
            b,
            out,
            stride_x[1],
            stride_x[2],
            stride_w[0],
            stride_w[1],
            stride_o[1],
            stride_o[2],
        )
        return out

    if len(x_shape) == 2 and x_shape[0] == 128 and x_shape[1] == 128 and w.shape[0] == 128 and w.shape[1] == 128:
        out = torch.empty((128, 128), device=x.device, dtype=x.dtype)
        stride_x = x.stride()
        stride_w = w.stride()
        stride_o = out.stride()
        grid = lambda META: (triton.cdiv(128, META['BLOCK_N']), triton.cdiv(128, META['BLOCK_M']))
        _rect_linear_kernel[grid](
            x,
            w,
            b,
            out,
            stride_x[0],
            stride_x[1],
            stride_w[0],
            stride_w[1],
            stride_o[0],
            stride_o[1],
        )
        return out

    raise RuntimeError('Unsupported shape in shared_dispatch linear route')