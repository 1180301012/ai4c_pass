import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(conv2d, 2)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_P": 32}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_P": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_P": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_P": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_P": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_P": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_P": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_P": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_P": 512}, num_warps=8, num_stages=2),
    ],
    key=["N", "P"],
)
@triton.jit
def _conv1x1_flatten_17x160_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N,
    P,
    x_stride_n,
    x_stride_c,
    w_stride_o,
    w_stride_c,
    out_stride_n,
    out_stride_o,
    BLOCK_P: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_n = tl.program_id(1)

    p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p < P

    o16 = tl.arange(0, 16)
    o1 = tl.arange(0, 1)

    acc16 = tl.zeros((16, BLOCK_P), dtype=tl.float32)
    acc1 = tl.zeros((1, BLOCK_P), dtype=tl.float32)
    x_base = x_ptr + pid_n * x_stride_n + p[None, :]

    for k0 in range(0, 160, 32):
        k = k0 + tl.arange(0, 32)
        x = tl.load(
            x_base + k[:, None] * x_stride_c,
            mask=p_mask[None, :],
            other=0.0,
        )
        w16 = tl.load(w_ptr + o16[:, None] * w_stride_o + k[None, :] * w_stride_c)
        w1 = tl.load(w_ptr + (16 + o1)[:, None] * w_stride_o + k[None, :] * w_stride_c)
        acc16 += tl.dot(w16, x)
        acc1 += tl.dot(w1, x)

    acc16 += tl.load(b_ptr + o16)[:, None]
    acc1 += tl.load(b_ptr + 16 + o1)[:, None]

    out_base = out_ptr + pid_n * out_stride_n + p[None, :]
    tl.store(out_base + o16[:, None] * out_stride_o, acc16, mask=p_mask[None, :])
    tl.store(out_base + (16 + o1)[:, None] * out_stride_o, acc1, mask=p_mask[None, :])


@torch.fx.wrap
def _fused_conv2d_flatten_17x160(bias, weight, x):
    n = x.shape[0]
    p = x.shape[2] * x.shape[3]

    out = torch.empty((n, 17, p), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(p, meta["BLOCK_P"]), n)
    _conv1x1_flatten_17x160_kernel[grid](
        x,
        weight,
        bias,
        out,
        n,
        p,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out


def replacement_func():
    return _fused_conv2d_flatten_17x160