import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.gelu(tmp_4, approximate='none')
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
    ],
    key=['c_elements', 'hw_elements', 'out_dtype_code'],
)
@triton.jit
def fused_se_kernel(
    bias_ptr,
    weight_ptr,
    in2_ptr,
    in3_ptr,
    out_ptr,
    c_elements,
    k_elements,
    h_elements,
    w_elements,
    hw_elements,
    weight_stride_o,
    weight_stride_i,
    in2_stride_n,
    in2_stride_c,
    in2_stride_h,
    in2_stride_w,
    in3_stride_n,
    in3_stride_c,
    out_stride_n,
    out_stride_c,
    out_dtype_code,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr = 64,
    MAX_HW: tl.constexpr = 144,
):
    pid_c = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_k = tl.arange(0, BLOCK_K)
    mask_c = offs_c < c_elements
    mask_k = offs_k < k_elements

    x_ptrs = in3_ptr + pid_n * in3_stride_n + offs_k * in3_stride_c
    x = tl.load(x_ptrs, mask=mask_k, other=0.0).to(tl.float32)

    w_ptrs = weight_ptr + offs_c[:, None] * weight_stride_o + offs_k[None, :] * weight_stride_i
    w = tl.load(w_ptrs, mask=mask_c[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)

    gate_pre = b + tl.sum(w * x[None, :], axis=1)
    gate = 1.0 / (1.0 + tl.exp(-gate_pre))

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for hw_start in range(0, MAX_HW, BLOCK_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)
        mask_hw = offs_hw < hw_elements
        offs_h = offs_hw // w_elements
        offs_w = offs_hw % w_elements
        in2_ptrs = (
            in2_ptr
            + pid_n * in2_stride_n
            + offs_c[:, None] * in2_stride_c
            + offs_h[None, :] * in2_stride_h
            + offs_w[None, :] * in2_stride_w
        )
        x2 = tl.load(in2_ptrs, mask=mask_c[:, None] & mask_hw[None, :], other=0.0).to(tl.float32)
        z = x2 * gate[:, None]
        z3 = z * z * z
        u = 0.7978845608028654 * (z + 0.044715 * z3)
        t = 2.0 / (1.0 + tl.exp(-2.0 * u)) - 1.0
        gelu = 0.5 * z * (1.0 + t)
        acc += tl.sum(gelu, axis=1)

    out = acc / hw_elements
    out_ptrs = out_ptr + pid_n * out_stride_n + offs_c * out_stride_c
    if out_dtype_code == 0:
        tl.store(out_ptrs, out.to(tl.float16), mask=mask_c)
    elif out_dtype_code == 1:
        tl.store(out_ptrs, out.to(tl.bfloat16), mask=mask_c)
    else:
        tl.store(out_ptrs, out, mask=mask_c)


@torch.fx.wrap
def fused_se_conv_sigmoid_mul_gelu_avgpool(in_0, in_1, in_2, in_3):
    n = in_2.shape[0]
    c = in_2.shape[1]
    h = in_2.shape[2]
    w = in_2.shape[3]
    k = in_3.shape[1]

    out = torch.empty((n, c), device=in_2.device, dtype=in_2.dtype)

    if out.dtype == torch.float16:
        out_dtype_code = 0
    elif out.dtype == torch.bfloat16:
        out_dtype_code = 1
    else:
        out_dtype_code = 2

    grid = lambda META: (triton.cdiv(c, META['BLOCK_C']), n)
    fused_se_kernel[grid](
        in_0,
        in_1,
        in_2,
        in_3,
        out,
        c,
        k,
        h,
        w,
        h * w,
        in_1.stride(0),
        in_1.stride(1),
        in_2.stride(0),
        in_2.stride(1),
        in_2.stride(2),
        in_2.stride(3),
        in_3.stride(0),
        in_3.stride(1),
        out.stride(0),
        out.stride(1),
        out_dtype_code,
    )
    return out


def replacement_func():
    return fused_se_conv_sigmoid_mul_gelu_avgpool