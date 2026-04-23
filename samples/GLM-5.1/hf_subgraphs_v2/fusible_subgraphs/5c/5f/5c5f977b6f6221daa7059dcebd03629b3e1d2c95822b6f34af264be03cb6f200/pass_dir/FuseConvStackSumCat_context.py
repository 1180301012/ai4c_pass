import torch
import triton
import triton.language as tl


def pattern(conv_result, cat_tensor):
    stacked = torch.stack([conv_result], dim=0)
    summed = stacked.sum(dim=0)
    result = torch.cat([summed, cat_tensor], 1)
    return result


def replacement_args(conv_result, cat_tensor):
    return (conv_result, cat_tensor)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 8192}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_HW': 8192}, num_warps=16, num_stages=1),
    ],
    key=['C_conv', 'C_cat', 'HW'],
)
@triton.jit
def cat_dim1_kernel(
    conv_ptr, cat_ptr, out_ptr,
    C_conv, C_cat, HW,
    stride_conv_n, stride_conv_c,
    stride_cat_n, stride_cat_c,
    stride_out_n, stride_out_c,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    dst_base = pid_n * stride_out_n + pid_c * stride_out_c

    if pid_c < C_conv:
        src_base = pid_n * stride_conv_n + pid_c * stride_conv_c
        src_ptr = conv_ptr
    else:
        c_cat = pid_c - C_conv
        src_base = pid_n * stride_cat_n + c_cat * stride_cat_c
        src_ptr = cat_ptr

    for hw_start in range(0, HW, BLOCK_HW):
        offsets = hw_start + tl.arange(0, BLOCK_HW)
        mask = offsets < HW
        val = tl.load(src_ptr + src_base + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + dst_base + offsets, val, mask=mask)


@torch.fx.wrap
def fused_cat(conv_result, cat_tensor):
    N = conv_result.shape[0]
    C_conv = conv_result.shape[1]
    C_cat = cat_tensor.shape[1]
    H = conv_result.shape[2]
    W = conv_result.shape[3]
    HW = H * W

    C_total = C_conv + C_cat

    out = torch.empty([N, C_total, H, W], dtype=conv_result.dtype, device=conv_result.device)

    stride_conv_n = conv_result.stride(0)
    stride_conv_c = conv_result.stride(1)
    stride_cat_n = cat_tensor.stride(0)
    stride_cat_c = cat_tensor.stride(1)
    stride_out_n = out.stride(0)
    stride_out_c = out.stride(1)

    grid = (N, C_total)

    cat_dim1_kernel[grid](
        conv_ptr=conv_result,
        cat_ptr=cat_tensor,
        out_ptr=out,
        C_conv=C_conv,
        C_cat=C_cat,
        HW=HW,
        stride_conv_n=stride_conv_n,
        stride_conv_c=stride_conv_c,
        stride_cat_n=stride_cat_n,
        stride_cat_c=stride_cat_c,
        stride_out_n=stride_out_n,
        stride_out_c=stride_out_c,
    )

    return out


def replacement_func():
    return fused_cat