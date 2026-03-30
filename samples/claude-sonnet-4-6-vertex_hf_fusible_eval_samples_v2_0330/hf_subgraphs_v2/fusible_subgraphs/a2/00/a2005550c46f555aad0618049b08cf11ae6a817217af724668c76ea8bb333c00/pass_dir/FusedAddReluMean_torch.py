import torch
import triton
import triton.language as tl

# ─────────────────────────────────────────────────────────────────────────────
# Variant B – uses torch.relu / torch.mean (torch.* function forms)
# Some FX tracers normalize F.relu → torch.relu and .mean() → torch.mean
# ─────────────────────────────────────────────────────────────────────────────

def pattern(tmp_4, in_5):
    tmp_5 = in_5 + tmp_4
    tmp_6 = torch.relu(tmp_5)
    tmp_7 = torch.mean(tmp_6, (2, 3), keepdim=True)
    return (tmp_6, tmp_7)


def replacement_args(tmp_4, in_5):
    return (tmp_4, in_5)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 2048}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 512},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=4,  num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def _v2_fused_add_relu_mean_kernel(
    x_ptr, res_ptr, out_ptr, mean_out_ptr,
    HW,
    DTYPE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid  = tl.program_id(0)
    base = pid * HW
    acc  = 0.0

    for block_start in range(0, HW, BLOCK_HW):
        idx  = block_start + tl.arange(0, BLOCK_HW)
        mask = idx < HW
        x   = tl.load(x_ptr  + base + idx, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(res_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y   = tl.maximum(x + res, 0.0)
        tl.store(out_ptr + base + idx, y.to(DTYPE), mask=mask)
        y_valid = tl.where(mask, y, tl.zeros_like(y))
        acc = acc + tl.sum(y_valid, axis=0)

    tl.store(mean_out_ptr + pid, (acc / HW).to(DTYPE))


_DTYPE_MAP_V2 = {
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32:  tl.float32,
}


@torch.fx.wrap
def v2_fused_add_relu_mean(tmp_4, in_5):
    N, C, H, W = tmp_4.shape
    HW = H * W
    out      = torch.empty_like(tmp_4)
    mean_out = torch.empty((N, C, 1, 1), dtype=tmp_4.dtype, device=tmp_4.device)
    DTYPE = _DTYPE_MAP_V2[tmp_4.dtype]
    _v2_fused_add_relu_mean_kernel[(N * C,)](
        tmp_4, in_5, out, mean_out.view(N * C), HW, DTYPE,
    )
    return (out, mean_out)


def replacement_func():
    return v2_fused_add_relu_mean