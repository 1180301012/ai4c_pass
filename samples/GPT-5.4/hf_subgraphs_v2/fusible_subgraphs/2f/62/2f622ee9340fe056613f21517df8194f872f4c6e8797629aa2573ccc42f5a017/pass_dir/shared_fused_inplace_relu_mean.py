import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_HW": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_HW": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_HW": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 512}, num_warps=8, num_stages=2),
    ],
    key=["HW"],
)
@triton.jit
def fused_inplace_relu_mean_contig_kernel(
    x_ptr,
    mean_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_HW)
    base = pid * HW
    acc = tl.zeros((), dtype=tl.float32)

    for start in range(0, HW, BLOCK_HW):
        idx = start + offs
        mask = idx < HW
        ptrs = x_ptr + base + idx
        x = tl.load(ptrs, mask=mask, other=0.0)
        y = tl.maximum(x, 0)
        tl.store(ptrs, y, mask=mask)
        acc += tl.sum(y.to(tl.float32), axis=0)

    tl.store(mean_ptr + pid, acc / HW)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_HW": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_HW": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_HW": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 512}, num_warps=8, num_stages=2),
    ],
    key=["HW"],
)
@triton.jit
def fused_inplace_relu_mean_strided_kernel(
    x_ptr,
    mean_ptr,
    C,
    W,
    x_s0,
    x_s1,
    x_s2,
    x_s3,
    HW,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    offs = tl.arange(0, BLOCK_HW)
    acc = tl.zeros((), dtype=tl.float32)

    for start in range(0, HW, BLOCK_HW):
        idx = start + offs
        mask = idx < HW
        h = idx // W
        w = idx % W
        ptrs = x_ptr + n * x_s0 + c * x_s1 + h * x_s2 + w * x_s3
        x = tl.load(ptrs, mask=mask, other=0.0)
        y = tl.maximum(x, 0)
        tl.store(ptrs, y, mask=mask)
        acc += tl.sum(y.to(tl.float32), axis=0)

    tl.store(mean_ptr + pid, acc / HW)


@torch.fx.wrap
def fused_inplace_relu_mean_graph(x):
    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    hw = h * w
    mean_out = torch.empty((n, c, 1, 1), device=x.device, dtype=x.dtype)

    strides = x.stride()
    is_contig_nchw = (
        strides[3] == 1
        and strides[2] == w
        and strides[1] == h * w
        and strides[0] == c * h * w
    )

    grid = (n * c,)
    if is_contig_nchw:
        fused_inplace_relu_mean_contig_kernel[grid](
            x_ptr=x,
            mean_ptr=mean_out,
            HW=hw,
        )
    else:
        fused_inplace_relu_mean_strided_kernel[grid](
            x_ptr=x,
            mean_ptr=mean_out,
            C=c,
            W=w,
            x_s0=strides[0],
            x_s1=strides[1],
            x_s2=strides[2],
            x_s3=strides[3],
            HW=hw,
        )
    return (x, mean_out)


def replacement_func():
    return fused_inplace_relu_mean_graph