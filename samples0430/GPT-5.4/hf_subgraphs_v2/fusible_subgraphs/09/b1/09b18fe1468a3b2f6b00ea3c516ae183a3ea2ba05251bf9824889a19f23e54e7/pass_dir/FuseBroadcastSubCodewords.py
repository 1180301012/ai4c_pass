import torch
import triton
import triton.language as tl


def pattern(in_0, in_4):
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    tmp_10 = tmp_8 - tmp_6
    return tmp_10


def replacement_args(in_0, in_4):
    return (in_0, in_4)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_K': 256}, num_warps=8, num_stages=4),
    ],
    key=[]
)
@triton.jit
def _broadcast_sub_kernel(
    x_ptr,
    code_ptr,
    out_ptr,
    x_stride_n,
    x_stride_k,
    code_stride_c,
    code_stride_k,
    out_stride_n,
    out_stride_c,
    out_stride_k,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_kb = tl.program_id(1)

    offs_c = tl.arange(0, 32)
    offs_k = pid_kb * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < 512
    mask = mask_k[None, :]

    x = tl.load(x_ptr + pid_n * x_stride_n + offs_k * x_stride_k, mask=mask_k, other=0.0)
    code = tl.load(code_ptr + offs_c[:, None] * code_stride_c + offs_k[None, :] * code_stride_k, mask=mask, other=0.0)
    out = x[None, :] - code

    out_ptrs = out_ptr + pid_n * out_stride_n + offs_c[:, None] * out_stride_c + offs_k[None, :] * out_stride_k
    tl.store(out_ptrs, out, mask=mask)


@torch.fx.wrap
def broadcast_sub_codewords_triton(in_0, in_4):
    n = in_4.shape[1]
    k = in_4.shape[2]
    out = torch.empty((1, n, 32, k), device=in_4.device, dtype=in_4.dtype)

    grid = lambda META: (n, triton.cdiv(k, META['BLOCK_K']))
    _broadcast_sub_kernel[grid](
        in_4,
        in_0,
        out,
        in_4.stride()[1],
        in_4.stride()[2],
        in_0.stride()[0],
        in_0.stride()[1],
        out.stride()[1],
        out.stride()[2],
        out.stride()[3],
    )
    return out



def replacement_func():
    return broadcast_sub_codewords_triton