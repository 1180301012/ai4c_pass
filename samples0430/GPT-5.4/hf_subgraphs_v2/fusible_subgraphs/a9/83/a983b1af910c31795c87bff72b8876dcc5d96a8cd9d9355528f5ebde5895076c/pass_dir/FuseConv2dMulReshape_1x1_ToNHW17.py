import torch
import triton
import triton.language as tl


# Pattern matching function
# Must mirror the original graph structure exactly.
def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d * 1.0
    tmp_4 = tmp_3.reshape(-1, 17, 4096)
    return tmp_4


# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 64, 'BLOCK_C': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_P': 128, 'BLOCK_C': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_P': 256, 'BLOCK_C': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_P': 64, 'BLOCK_C': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_P': 128, 'BLOCK_C': 64}, num_warps=8, num_stages=2),
    ],
    key=['P', 'C'],
)
@triton.jit
def conv1x1_reshape_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    P,
    stride_x_n,
    stride_x_c,
    stride_x_h,
    stride_x_w,
    stride_w_o,
    stride_w_c,
    stride_out_n,
    stride_out_o,
    stride_out_p,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_no = tl.program_id(1)

    o = pid_no % 17
    n = pid_no // 17

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = offs_p < P

    offs_h = offs_p // W
    offs_w = offs_p - offs_h * W

    acc = tl.zeros((BLOCK_P,), dtype=tl.float32)

    c_start = 0
    while c_start < C:
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        x_ptrs = x_ptr + (
            n * stride_x_n
            + offs_c[:, None] * stride_x_c
            + offs_h[None, :] * stride_x_h
            + offs_w[None, :] * stride_x_w
        )
        w_ptrs = w_ptr + o * stride_w_o + offs_c * stride_w_c

        x = tl.load(x_ptrs, mask=mask_c[:, None] & mask_p[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_c, other=0.0)
        acc += tl.sum(x * w[:, None], axis=0)
        c_start += BLOCK_C

    bias = tl.load(b_ptr + o)
    acc += bias.to(tl.float32)

    out_ptrs = out_ptr + n * stride_out_n + o * stride_out_o + offs_p * stride_out_p
    tl.store(out_ptrs, acc, mask=mask_p)


# Kernel wrapper
# Intentionally top-level and stable so replacement_func() is deterministic.
def fused_conv2d_mul_reshape(in_0, in_1, in_2):
    n = in_2.shape[0]
    c = in_2.shape[1]
    h = in_2.shape[2]
    w = in_2.shape[3]
    p = h * w

    out = torch.empty((n, 17, p), device=in_2.device, dtype=in_2.dtype)

    grid = lambda meta: (
        triton.cdiv(p, meta['BLOCK_P']),
        n * 17,
    )

    conv1x1_reshape_kernel[grid](
        in_2,
        in_1,
        in_0,
        out,
        n,
        c,
        h,
        w,
        p,
        in_2.stride(0),
        in_2.stride(1),
        in_2.stride(2),
        in_2.stride(3),
        in_1.stride(0),
        in_1.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    return out


# Replacement function (NO arguments)
def replacement_func():
    return fused_conv2d_mul_reshape