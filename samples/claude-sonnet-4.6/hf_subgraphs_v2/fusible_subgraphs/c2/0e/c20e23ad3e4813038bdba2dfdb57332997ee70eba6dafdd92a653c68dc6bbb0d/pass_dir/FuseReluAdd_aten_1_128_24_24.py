import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fallback pattern (aten ops, no interpolate):
#   aten.relu_.default(conv2d_out)  +  aten.add.Tensor(in_2, relu_out)
# The downstream interpolate(24→24) is identity so correctness is preserved.
# ---------------------------------------------------------------------------
def pattern(conv2d_out, in_2):
    tmp_3 = torch.ops.aten.relu_.default(conv2d_out)
    tmp_4 = torch.ops.aten.add.Tensor(in_2, tmp_3)
    return tmp_4


def replacement_args(conv2d_out, in_2):
    return (conv2d_out, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: fused relu(x) + y
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
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
    relu_x = tl.maximum(x, 0.0)
    out = relu_x + y
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_relu_add_aten(conv2d_out, in_2):
    N = conv2d_out.numel()          # 1 * 128 * 24 * 24 = 73 728
    out = torch.empty_like(conv2d_out)
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _relu_add_kernel_aten[grid](
        conv2d_out,
        in_2,
        out,
        N,
    )
    return out


def replacement_func():
    return fused_relu_add_aten