import operator
import torch
import torch.fx.proxy as fx_proxy
import triton
import triton.language as tl

# Monkey-patch torch.fx.Proxy.__iadd__ so that `+=` in the pattern function
# creates an 'iadd' FX node (matching the dynamo-traced target graph).
def _proxy_iadd(self, other):
    return self.tracer.create_proxy('call_function', operator.iadd, (self, other), {})

fx_proxy.Proxy.__iadd__ = _proxy_iadd


def pattern(in_0, in_1, in_2):
    tmp_0 = in_0 / 8.0
    tmp_0 += in_2          # creates 'iadd' node via patched __iadd__
    tmp_2 = tmp_0 + in_1
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N_ELEM'],
)
@triton.jit
def fused_scale_add_kernel(
    in0_ptr, in1_ptr, in2_ptr, out_ptr,
    N_ELEM, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEM

    x0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(in2_ptr + offsets, mask=mask, other=0.0)
    # in_1 shape is [batch, 1, 1, n_cols]; broadcast over dims 1 and 2
    in1_offsets = (offsets // n_cols) * n_cols + (offsets % n_cols)
    x1 = tl.load(in1_ptr + in1_offsets, mask=mask, other=0.0)

    out = x0 / 8.0 + x2 + x1
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_scale_add(in_0, in_1, in_2):
    N_ELEM = in_0.numel()
    out_shape = list(in_0.shape)
    out = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    n_cols = in_0.shape[-1]

    grid = lambda meta: ((N_ELEM + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_scale_add_kernel[grid](
        in_0, in_1, in_2, out,
        N_ELEM, n_cols,
    )

    return out


def replacement_func():
    return fused_scale_add