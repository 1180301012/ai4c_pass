"""
Pattern: 0 + in_1; tmp_0 += in_0 (iadd); mean((2,3), keepdim=True)
Fuses two-input addition with spatial mean reduction into one Triton kernel.
"""
import operator
import torch
import torch.fx
import triton
import triton.language as tl

# Monkey-patch Proxy so that += produces an iadd node, matching the target graph.
if not hasattr(torch.fx.Proxy, '__iadd__') or \
        getattr(torch.fx.Proxy.__iadd__, '_is_iadd_patched', False) is False:
    def _proxy_iadd(self, other):
        return self.tracer.create_proxy('call_function', operator.iadd, (self, other), {})
    _proxy_iadd._is_iadd_patched = True
    torch.fx.Proxy.__iadd__ = _proxy_iadd


def pattern(in_0, in_1):
    tmp_0 = 0 + in_1
    tmp_0 += in_0
    tmp_2 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_2)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32},  num_warps=1),
        triton.Config({'BLOCK_HW': 64},  num_warps=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _fused_add_mean_2input_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    mean_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    """Each program handles one (N, C) slice and tiles over HW.
    Computes: out = in0 + in1, mean = out.mean(spatial)
    """
    pid = tl.program_id(0)
    base = pid * HW

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for start in range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        mem_offs = base + offs

        a = tl.load(in0_ptr + mem_offs, mask=mask, other=0.0)
        b = tl.load(in1_ptr + mem_offs, mask=mask, other=0.0)
        val = a + b

        tl.store(out_ptr + mem_offs, val, mask=mask)
        acc += tl.where(mask, val.to(tl.float32), tl.zeros([BLOCK_HW], dtype=tl.float32))

    mean_val = tl.sum(acc) / HW
    tl.store(mean_ptr + pid, mean_val)


@torch.fx.wrap
def _run_fused_add_mean_2input(in_0, in_1):
    """Fused add + spatial mean.
    Uses Triton for large NC (bandwidth savings outweigh launch overhead);
    falls back to PyTorch for small NC or scalar inputs.
    """
    if isinstance(in_0, torch.Tensor):
        ref = in_0
    else:
        ref = in_1

    N, C, H, W = ref.shape
    HW = H * W
    NC = N * C

    if isinstance(in_0, torch.Tensor) and isinstance(in_1, torch.Tensor) and NC >= 4096:
        # Large NC: Triton fused kernel (1 memory pass for add + mean).
        in_0_c = in_0.contiguous()
        in_1_c = in_1.contiguous()
        out = torch.empty_like(in_0_c)
        mean_buf = torch.empty(NC, dtype=torch.float32, device=in_0.device)
        _fused_add_mean_2input_kernel[(NC,)](
            in_0_c, in_1_c, out, mean_buf, HW,
        )
        return out, mean_buf.view(N, C, 1, 1).to(in_0.dtype)
    else:
        # Small NC or scalar input: use PyTorch (avoids Triton launch overhead).
        out = in_0 + in_1 if isinstance(in_0, torch.Tensor) else in_1 + in_0
        return out, out.mean((2, 3), keepdim=True)


# NOT wrapped: FX traces this so result[0] and result[1] become two
# separate getitem nodes → len(copied_returning_nodes) == 2.
def fused_add_mean_2input(in_0, in_1):
    result = _run_fused_add_mean_2input(in_0, in_1)
    return result[0], result[1]


def replacement_func():
    return fused_add_mean_2input