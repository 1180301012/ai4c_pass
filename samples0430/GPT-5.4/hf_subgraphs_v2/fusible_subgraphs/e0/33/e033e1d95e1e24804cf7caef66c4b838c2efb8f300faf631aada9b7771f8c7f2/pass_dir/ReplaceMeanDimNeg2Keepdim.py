import torch
import triton
import triton.language as tl


@triton.jit
def mean_dim1_kernel(
    x_ptr,
    out_ptr,
    B,
    R,
    C,
    stride_xb,
    stride_xr,
    stride_xc,
    stride_ob,
    stride_oc,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % C
    b = pid // C

    if b >= B:
        return

    acc = tl.zeros((), dtype=tl.float32)
    for r0 in tl.static_range(0, 4096, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R
        vals = tl.load(
            x_ptr + b * stride_xb + offs_r * stride_xr + c * stride_xc,
            mask=mask_r,
            other=0.0,
        )
        acc += tl.sum(vals.to(tl.float32), axis=0)

    acc = acc / R
    tl.store(out_ptr + b * stride_ob + c * stride_oc, acc)


@torch.fx.wrap
def triton_mean_dim_neg2_keepdim(x):
    b = x.shape[0]
    r = x.shape[1]
    c = x.shape[2]
    out = torch.empty((b, 1, c), device=x.device, dtype=x.dtype)
    grid = (b * c,)
    mean_dim1_kernel[grid](
        x,
        out,
        b,
        r,
        c,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(2),
        BLOCK_R=256,
        num_warps=4,
        num_stages=4,
    )
    return out


def pattern(x):
    return x.mean(dim=-2, keepdim=True)


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_mean_dim_neg2_keepdim