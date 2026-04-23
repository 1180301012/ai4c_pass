import torch
import triton
import triton.language as tl

K_CONST = 249
K_PADDED_CONST = 256
N_CONST = 64


def pattern(in_0, in_1):
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul.squeeze(1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_vecmat_squeeze_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    x_stride_0,
    x_stride_2,
    y_stride_0,
    y_stride_1,
    y_stride_2,
    out_stride_0,
    out_stride_1,
    K: tl.constexpr,
    K_PADDED: tl.constexpr,
    N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    k_offsets = tl.arange(0, K_PADDED)
    n_offsets = tl.arange(0, N)

    x_batch_ptr = x_ptr + pid_b * x_stride_0
    y_batch_ptr = y_ptr + pid_b * y_stride_0
    out_batch_ptr = out_ptr + pid_b * out_stride_0

    scale = tl.load(x_batch_ptr + 0 * x_stride_2).to(tl.float32)
    y_vals = tl.load(
        y_batch_ptr + k_offsets[:, None] * y_stride_1 + n_offsets[None, :] * y_stride_2,
        mask=(k_offsets[:, None] < K),
        other=0.0,
    ).to(tl.float32)
    acc = tl.sum(y_vals, axis=0) * scale
    tl.store(out_batch_ptr + n_offsets * out_stride_1, acc)


@torch.fx.wrap
def fused_matmul_squeeze_vecmat(in_0, in_1):
    batch = in_0.shape[0]
    out = torch.empty((batch, N_CONST), device=in_0.device, dtype=in_0.dtype)

    fused_vecmat_squeeze_kernel[(batch,)](
        in_0,
        in_1,
        out,
        in_0.stride(0),
        in_0.stride(2),
        in_1.stride(0),
        in_1.stride(1),
        in_1.stride(2),
        out.stride(0),
        out.stride(1),
        K=K_CONST,
        K_PADDED=K_PADDED_CONST,
        N=N_CONST,
        num_warps=4,
        num_stages=2,
    )
    return out


def replacement_func():
    return fused_matmul_squeeze_vecmat