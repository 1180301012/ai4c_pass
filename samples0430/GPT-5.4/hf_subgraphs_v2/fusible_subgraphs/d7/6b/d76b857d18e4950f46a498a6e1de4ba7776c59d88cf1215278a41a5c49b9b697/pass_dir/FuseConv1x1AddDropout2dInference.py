import torch
import triton
import triton.language as tl


# Pattern matching function
# Must mirror model.py exactly.
def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = in_4 + in_3
    tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    return (tmp_4, tmp_2)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32, 'BLOCK_CO': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32, 'BLOCK_CO': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64, 'BLOCK_CO': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32, 'BLOCK_CO': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64, 'BLOCK_CO': 32}, num_warps=8, num_stages=4),
    ],
    key=['total_rows', 'c_in', 'c_out'],
)
@triton.jit
def conv1x1_nchw_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    total_rows,
    h,
    w,
    c_in,
    c_out,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_woc,
    stride_wic,
    stride_on,
    stride_oc,
    stride_oh,
    stride_ow,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_co = tl.arange(0, BLOCK_CO)

    hw = offs_m % (h * w)
    n_idx = offs_m // (h * w)
    h_idx = hw // w
    w_idx = hw % w

    acc = tl.zeros((BLOCK_M, BLOCK_CO), dtype=tl.float32)

    for k0 in range(0, c_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x_ptrs = (
            x_ptr
            + n_idx[:, None] * stride_xn
            + offs_k[None, :] * stride_xc
            + h_idx[:, None] * stride_xh
            + w_idx[:, None] * stride_xw
        )
        x_mask = (offs_m[:, None] < total_rows) & (offs_k[None, :] < c_in)
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = w_ptr + offs_co[None, :] * stride_woc + offs_k[:, None] * stride_wic
        w_mask = (offs_co[None, :] < c_out) & (offs_k[:, None] < c_in)
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    bias = tl.load(b_ptr + offs_co, mask=offs_co < c_out, other=0.0)
    acc += bias[None, :]

    out_ptrs = (
        out_ptr
        + n_idx[:, None] * stride_on
        + offs_co[None, :] * stride_oc
        + h_idx[:, None] * stride_oh
        + w_idx[:, None] * stride_ow
    )
    out_mask = (offs_m[:, None] < total_rows) & (offs_co[None, :] < c_out)
    tl.store(out_ptrs, acc, mask=out_mask)


@torch.fx.wrap
def triton_add_wrapper(x_skip_0, x_skip_1):
    add_out = torch.empty_like(x_skip_0)
    n_elements = add_out.numel()
    add_grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[add_grid](
        x_skip_0,
        x_skip_1,
        add_out,
        n_elements,
    )
    return add_out


@torch.fx.wrap
def triton_conv1x1_wrapper(bias, weight, inp):
    batch = inp.shape[0]
    height = inp.shape[2]
    width = inp.shape[3]
    c_in = inp.shape[1]
    c_out = weight.shape[0]
    total_rows = batch * height * width

    conv_out = torch.empty((batch, c_out, height, width), device=inp.device, dtype=inp.dtype)
    conv_grid = lambda meta: (triton.cdiv(total_rows, meta['BLOCK_M']),)
    conv1x1_nchw_kernel[conv_grid](
        inp,
        weight,
        bias,
        conv_out,
        total_rows,
        height,
        width,
        c_in,
        c_out,
        inp.stride(0),
        inp.stride(1),
        inp.stride(2),
        inp.stride(3),
        weight.stride(0),
        weight.stride(1),
        conv_out.stride(0),
        conv_out.stride(1),
        conv_out.stride(2),
        conv_out.stride(3),
    )
    return conv_out


def fused_cam_seg_infer(bias, weight, inp, x_skip_0, x_skip_1):
    add_out = triton_add_wrapper(x_skip_0, x_skip_1)
    conv_out = triton_conv1x1_wrapper(bias, weight, inp)
    return (add_out, conv_out)


def replacement_func():
    return fused_cam_seg_infer