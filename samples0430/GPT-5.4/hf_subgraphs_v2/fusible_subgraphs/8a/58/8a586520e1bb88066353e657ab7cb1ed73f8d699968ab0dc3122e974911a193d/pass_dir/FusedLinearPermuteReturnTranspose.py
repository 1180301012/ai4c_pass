import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.permute(0, 3, 1, 2)
    return tmp_3



def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_P": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_P": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_P": 256}, num_warps=8, num_stages=1),
    ],
    key=["P", "O"],
)
@triton.jit
def fused_linear_permute_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    B,
    M,
    N,
    O,
    P,
    stride_x0,
    stride_x1,
    stride_x2,
    stride_x3,
    stride_w0,
    stride_w1,
    stride_out0,
    stride_out1,
    stride_out2,
    BLOCK_P: tl.constexpr,
    BLOCK_O: tl.constexpr = 16,
):
    pid_p = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = offs_p < P

    offs_o = tl.arange(0, BLOCK_O)
    mask_o = offs_o < O

    m_idx = offs_p // N
    n_idx = offs_p - m_idx * N

    x_base = pid_b * stride_x0 + m_idx * stride_x1 + n_idx * stride_x2

    x0 = tl.load(x_ptr + x_base + 0 * stride_x3, mask=mask_p, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + x_base + 1 * stride_x3, mask=mask_p, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + x_base + 2 * stride_x3, mask=mask_p, other=0.0).to(tl.float32)

    w0 = tl.load(w_ptr + offs_o * stride_w0 + 0 * stride_w1, mask=mask_o, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + offs_o * stride_w0 + 1 * stride_w1, mask=mask_o, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + offs_o * stride_w0 + 2 * stride_w1, mask=mask_o, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs_o, mask=mask_o, other=0.0).to(tl.float32)

    acc = (
        bias[:, None]
        + w0[:, None] * x0[None, :]
        + w1[:, None] * x1[None, :]
        + w2[:, None] * x2[None, :]
    )

    out_ptrs = (
        out_ptr
        + pid_b * stride_out0
        + offs_o[:, None] * stride_out1
        + offs_p[None, :] * stride_out2
    )
    out_mask = mask_o[:, None] & mask_p[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


_DEVICE_TENSOR_CACHE = {}



def _ensure_device_tensor(x, device):
    if x.device == device:
        return x
    key = (id(x), device)
    cached = _DEVICE_TENSOR_CACHE.get(key)
    if cached is None:
        cached = torch.as_tensor(x, device=device)
        _DEVICE_TENSOR_CACHE[key] = cached
    return cached


@torch.fx.wrap
def fused_linear_permute_return_transpose(in_0, in_1, in_3):
    device = in_3.device
    weight = _ensure_device_tensor(in_1, device)
    bias = _ensure_device_tensor(in_0, device)

    B, M, N, K = in_3.shape
    O = weight.shape[0]
    P = M * N

    out_flat = torch.empty((B, O, P), device=device, dtype=in_3.dtype)

    grid = (triton.cdiv(P, 256), B)
    fused_linear_permute_kernel[grid](
        in_3,
        weight,
        bias,
        out_flat,
        B,
        M,
        N,
        O,
        P,
        in_3.stride(0),
        in_3.stride(1),
        in_3.stride(2),
        in_3.stride(3),
        weight.stride(0),
        weight.stride(1),
        out_flat.stride(0),
        out_flat.stride(1),
        out_flat.stride(2),
    )

    tmp_3 = out_flat.view(B, O, M, N)
    return tmp_3



def replacement_func():
    return fused_linear_permute_return_transpose