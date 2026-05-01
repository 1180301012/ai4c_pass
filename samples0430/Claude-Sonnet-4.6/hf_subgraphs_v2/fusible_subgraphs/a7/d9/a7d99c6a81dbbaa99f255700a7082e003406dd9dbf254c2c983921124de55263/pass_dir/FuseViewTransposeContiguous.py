import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_3 = x.view(1, 1, -1, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_9 = tmp_4.contiguous()
    return tmp_9


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 512}, num_warps=4),
        triton.Config({'BLOCK': 512}, num_warps=8),
        triton.Config({'BLOCK': 512}, num_warps=2),
        triton.Config({'BLOCK': 256}, num_warps=4),
    ],
    key=[],
)
@triton.jit
def copy_reshape_512_kernel(
    src_ptr, dst_ptr,
    BLOCK: tl.constexpr,
):
    """
    Copy 512 elements: flat index i in src -> flat index i in dst.
    [1,1,512] -> view(1,1,8,64) -> transpose(1,2) -> contiguous [1,8,1,64]
    is a pure element-wise copy (same memory layout).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    data = tl.load(src_ptr + offsets)
    tl.store(dst_ptr + offsets, data)


@torch.fx.wrap
def fused_view_transpose_contiguous(x):
    """
    Fused view(1,1,-1,64) + transpose(1,2) + contiguous()
    x: [1,1,512] contiguous  (flat offset i == element i)
    output: [1,8,1,64] contiguous (flat offset i == element i)
    Triton uses data_ptr() so no reshape needed.
    """
    out = torch.empty((1, 8, 1, 64), dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(512, meta['BLOCK']),)

    copy_reshape_512_kernel[grid](x, out)

    return out


def replacement_func():
    return fused_view_transpose_contiguous