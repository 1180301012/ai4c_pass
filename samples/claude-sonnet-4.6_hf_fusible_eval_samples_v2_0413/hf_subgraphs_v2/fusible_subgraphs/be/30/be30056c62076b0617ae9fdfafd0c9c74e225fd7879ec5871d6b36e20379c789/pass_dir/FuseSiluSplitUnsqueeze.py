import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: apply SiLU in-place over a flat 1-D view of the tensor.
# We upcast to fp32 for the sigmoid computation to avoid precision loss with
# fp16/bf16, then store back in the original dtype.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 8192}),
    ],
    key=['n_elements'],
)
@triton.jit
def _silu_inplace_kernel(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    # upcast to fp32 for accurate sigmoid
    xf = x.to(tl.float32)
    out = xf * tl.sigmoid(xf)
    tl.store(x_ptr + offset, out.to(x.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Kernel wrapper: SiLU in-place → return split views + in_0 view
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_silu_split_unsqueeze(in_0, in_1):
    n = in_1.numel()
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _silu_inplace_kernel[grid](in_1, n)

    # All three ops below are zero-copy views on the now-SiLU'd buffer
    tmp_3 = in_1[:, :, :512]          # [B, 17, 512]
    tmp_4 = in_1[:, :, 512:1024]      # [B, 17, 512]
    tmp_5 = in_1[:, :, 1024:]         # [B, 17, 128]
    tmp_6 = tmp_5.unsqueeze(2)        # [B, 17, 1, 128]
    tmp_7 = in_0[None, None, :]       # [1, 1, 2, 128]
    return (tmp_7, tmp_3, tmp_6, tmp_4)


# ---------------------------------------------------------------------------
# Pattern: mirrors model.py exactly
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
    split = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    tmp_3 = split[0]
    tmp_4 = split[1]
    tmp_5 = split[2]
    tmp_6 = tmp_5.unsqueeze(2)
    tmp_7 = in_0[(None, None, slice(None, None, None))]
    return (tmp_7, tmp_3, tmp_6, tmp_4)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_silu_split_unsqueeze