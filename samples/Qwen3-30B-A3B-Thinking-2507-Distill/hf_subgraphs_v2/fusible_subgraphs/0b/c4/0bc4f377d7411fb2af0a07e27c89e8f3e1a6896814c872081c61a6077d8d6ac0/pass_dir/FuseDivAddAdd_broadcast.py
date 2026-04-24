import operator
import torch
import torch.fx as _fx
import triton
import triton.language as tl

# Force FX Proxy to create iadd nodes for +=, so the pattern matches the
# target graph's call_function[operator.iadd] nodes.
def _proxy_iadd(self, other):
    return self.tracer.create_proxy('call_function', operator.iadd, (self, other), {})

_fx.Proxy.__iadd__ = _proxy_iadd


# Fused kernel: out = in_0 / 8.0 + in_2 + in_1
#  in_0, in_2: [B, H, S, S]  — flat idx = b*SPATIAL + sp, SPATIAL = H*S*S
#  in_1:        [B, 1, 1, S]  — flat idx = b*S + k
#  out:         [B, H, S, S]
#
# 2-D grid: pid(0)=batch, pid(1)=spatial block.
# Avoids the non-power-of-2 div inside the kernel.
@triton.jit
def _fused_div_add_add_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    SPATIAL: tl.constexpr,   # H * S * S = 588
    S: tl.constexpr,          # S = 7
    BLOCK_SIZE: tl.constexpr,
):
    b   = tl.program_id(0)          # batch (0 or 1)
    pid = tl.program_id(1)          # spatial block index

    sp_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = sp_offsets < SPATIAL

    global_offsets = b * SPATIAL + sp_offsets

    x = tl.load(in_0_ptr + global_offsets, mask=mask, other=0.0)
    z = tl.load(in_2_ptr + global_offsets, mask=mask, other=0.0)

    k = sp_offsets % S              # k index in [0, S)
    y = tl.load(in_1_ptr + b * S + k, mask=mask, other=0.0)

    out = x * 0.125 + z + y
    tl.store(out_ptr + global_offsets, out, mask=mask)


@torch.fx.wrap
def fused_div_add_add(in_0, in_1, in_2):
    B       = in_0.shape[0]   # 2
    H       = in_0.shape[1]   # 12
    S       = in_0.shape[3]   # 7
    SPATIAL = H * S * S       # 588

    out = torch.empty_like(in_0)

    # BLOCK_SIZE=1024: covers SPATIAL=588 in 1 spatial block per batch.
    # 2 blocks total (one per batch), 32 warps each → 64 warps total
    # → 2 SMs active, each at 100% warp occupancy (32 warps / 32 max).
    BLOCK_SIZE = 1024
    n_sp = (SPATIAL + BLOCK_SIZE - 1) // BLOCK_SIZE   # = 1

    _fused_div_add_add_kernel[(B, n_sp)](
        in_0, in_1, in_2, out,
        SPATIAL=SPATIAL,
        S=S,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )
    return out


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    tmp_0 = in_0 / 8.0
    tmp_0 += in_2
    tmp_2 = tmp_0 + in_1
    return (tmp_2,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_div_add_add