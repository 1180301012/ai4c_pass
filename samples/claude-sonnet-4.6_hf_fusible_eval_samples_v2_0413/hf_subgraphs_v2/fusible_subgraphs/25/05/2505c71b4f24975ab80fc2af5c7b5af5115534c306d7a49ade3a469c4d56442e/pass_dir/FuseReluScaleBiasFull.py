import torch
import triton
import triton.language as tl

# Register F.relu as an FX leaf so pattern tracer creates call_function(F.relu,...)
# matching the model's traced graph which uses torch.nn.functional.relu
torch.fx.wrap('torch.nn.functional.relu')


# Full fusion: relu(in_2) * in_1 + in_0
def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_relu_scale_bias_full_kernel(
    x_ptr, scale_ptr, bias_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x     = tl.load(x_ptr     + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr)
    bias  = tl.load(bias_ptr)

    # relu + scale + bias fused
    out = tl.maximum(x, 0.0) * scale + bias
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_relu_scale_bias_full(bias, scale, x):
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    fused_relu_scale_bias_full_kernel[grid](
        x_ptr=x,
        scale_ptr=scale,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
    )
    return out


def replacement_func():
    return fused_relu_scale_bias_full