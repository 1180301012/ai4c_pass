import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['n_out'],
)
@triton.jit
def _fused_gelu_pad_kernel_v3(
    in_ptr, out_ptr, n_in, n_out, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_out = offsets < n_out
    mask_in  = offsets < n_in
    x = tl.load(in_ptr + offsets, mask=mask_in, other=0.0)
    x_f32 = x.to(tl.float32)
    INV_SQRT2 = 0.7071067811865476
    gelu_f32 = 0.5 * x_f32 * (1.0 + tl.math.erf(x_f32 * INV_SQRT2))
    tl.store(out_ptr + offsets, gelu_f32.to(x.dtype), mask=mask_out)


@torch.fx.wrap
def _fused_gelu_reshape_pad_dispatch(in_0, _route=''):
    n_in = in_0.numel()
    n_out = 1 * 249 * 768
    out = torch.empty((1, 249, 768), dtype=in_0.dtype, device=in_0.device)
    grid = lambda meta: ((n_out + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _fused_gelu_pad_kernel_v3[grid](in_0, out, n_in, n_out)
    return out


# Pattern: F.gelu(approximate='none') + .reshape() + F.pad(0)
def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0.reshape(1, 124, 2, 768)
    tmp_2 = tmp_1.reshape(1, 248, 768)
    tmp_3 = torch.nn.functional.pad(tmp_2, (0, 0, 0, 1), 'constant', 0)
    return tmp_3


def replacement_args(in_0):
    return (in_0, 'v3')


def replacement_func():
    return _fused_gelu_reshape_pad_dispatch