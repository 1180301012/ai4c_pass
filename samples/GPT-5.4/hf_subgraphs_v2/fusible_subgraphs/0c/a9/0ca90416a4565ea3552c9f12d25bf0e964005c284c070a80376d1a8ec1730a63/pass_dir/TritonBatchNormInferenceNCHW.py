import torch
import triton
import triton.language as tl


# Pattern matching function
# Matches inference batch_norm exactly as it appears in the target graphs.
def pattern(in_0, in_1, in_2, in_3, in_4):
    return torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)


# Reorder args for a natural replacement signature: (x, mean, var, weight, bias)
def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_4, in_0, in_1, in_3, in_2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
    ],
    key=["n_elements", "HW", "C"],
)
@triton.jit
def bn_affine_apply_contiguous_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    n_elements,
    C,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    c = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + c, mask=mask, other=1.0)
    shift = tl.load(shift_ptr + c, mask=mask, other=0.0)
    y = x * scale + shift
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
    ],
    key=["n_elements", "C", "H", "W"],
)
@triton.jit
def bn_affine_apply_strided_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    n_elements,
    C,
    H,
    W,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    CHW,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    n = offsets // CHW
    rem0 = offsets % CHW
    c = rem0 // HW
    rem1 = rem0 % HW
    h = rem1 // W
    w = rem1 % W

    x_ptrs = x_ptr + n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
    out_ptrs = out_ptr + n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + c, mask=mask, other=1.0)
    shift = tl.load(shift_ptr + c, mask=mask, other=0.0)
    y = x * scale + shift
    tl.store(out_ptrs, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
    ],
    key=["C"],
)
@triton.jit
def bn_precompute_affine_kernel(
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    scale_ptr,
    shift_ptr,
    C,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = c < C

    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)

    inv_std = tl.rsqrt(var + EPS)
    scale = weight * inv_std
    shift = bias - mean * scale

    tl.store(scale_ptr + c, scale, mask=mask)
    tl.store(shift_ptr + c, shift, mask=mask)


@torch.fx.wrap
def triton_batch_norm_inference_nchw(x, running_mean, running_var, weight, bias):
    n, c, h, w = x.shape
    hw = h * w
    chw = c * hw
    n_elements = x.numel()

    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    scale = torch.empty((c,), device=x.device, dtype=torch.float32)
    shift = torch.empty((c,), device=x.device, dtype=torch.float32)

    precompute_grid = lambda META: (triton.cdiv(c, META["BLOCK_SIZE"]),)
    bn_precompute_affine_kernel[precompute_grid](
        mean_ptr=running_mean,
        var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        scale_ptr=scale,
        shift_ptr=shift,
        C=c,
        EPS=0.001,
    )

    if x.is_contiguous():
        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        bn_affine_apply_contiguous_kernel[grid](
            x_ptr=x,
            scale_ptr=scale,
            shift_ptr=shift,
            out_ptr=out,
            n_elements=n_elements,
            C=c,
            HW=hw,
        )
    else:
        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        bn_affine_apply_strided_kernel[grid](
            x_ptr=x,
            scale_ptr=scale,
            shift_ptr=shift,
            out_ptr=out,
            n_elements=n_elements,
            C=c,
            H=h,
            W=w,
            x_stride_n=x.stride(0),
            x_stride_c=x.stride(1),
            x_stride_h=x.stride(2),
            x_stride_w=x.stride(3),
            out_stride_n=out.stride(0),
            out_stride_c=out.stride(1),
            out_stride_h=out.stride(2),
            out_stride_w=out.stride(3),
            CHW=chw,
            HW=hw,
        )
    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return triton_batch_norm_inference_nchw