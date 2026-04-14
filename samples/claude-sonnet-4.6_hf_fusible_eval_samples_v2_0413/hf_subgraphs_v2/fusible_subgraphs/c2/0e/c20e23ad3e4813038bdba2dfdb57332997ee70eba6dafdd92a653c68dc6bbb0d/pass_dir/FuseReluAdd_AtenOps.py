import torch
import triton
import triton.language as tl

# Attempt with ATen-level ops — decomposed graphs often store nodes as
# torch.ops.aten.relu_.default / torch.ops.aten.add.Tensor
def pattern(conv2d_out, in_2):
    tmp_3 = torch.ops.aten.relu_.default(conv2d_out)
    tmp_4 = torch.ops.aten.add.Tensor(in_2, tmp_3)
    return tmp_4


def replacement_args(conv2d_out, in_2):
    return (conv2d_out, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _relu_add_kernel_aten(
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
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    out_f32 = tl.maximum(x_f32, 0.0) + y_f32
    tl.store(out_ptr + offsets, out_f32.to(tl.float16), mask=mask)


@torch.fx.wrap
def fused_relu_add_aten(conv2d_out, in_2):
    N = conv2d_out.numel()
    out = torch.empty_like(conv2d_out)
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _relu_add_kernel_aten[grid](
        x_ptr=conv2d_out,
        y_ptr=in_2,
        out_ptr=out,
        n_elements=N,
    )
    return out


def replacement_func():
    return fused_relu_add_aten