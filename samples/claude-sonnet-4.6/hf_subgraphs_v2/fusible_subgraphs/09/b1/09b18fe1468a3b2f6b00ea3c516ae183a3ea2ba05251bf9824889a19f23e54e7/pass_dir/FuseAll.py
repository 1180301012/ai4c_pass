import torch
import triton
import triton.language as tl


# ─── Pattern ────────────────────────────────────────────────────────────────
# Match chain 2 only: view + unsqueeze + expand + sub  (single output, confirmed working)

def pattern(in_0, in_4):
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    tmp_10 = tmp_8 - tmp_6
    return tmp_10


def replacement_args(in_0, in_4):
    return (in_0, in_4)


# ─── Kernel: Fused broadcast subtraction ─────────────────────────────────────
#   out[0,n,k,j] = in_4[0,n,j] - in_0[k,j]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_D': 256}, num_warps=4),
        triton.Config({'BLOCK_D': 512}, num_warps=2),
        triton.Config({'BLOCK_D': 512}, num_warps=4),
        triton.Config({'BLOCK_D': 512}, num_warps=8),
    ],
    key=['D'],
)
@triton.jit
def _broadcast_sub_kernel(
    in4_ptr, in0_ptr, out_ptr,
    N, K, D,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0)
    k = tl.program_id(1)
    for d_start in range(0, D, BLOCK_D):
        d_idx = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_idx < D
        # in4[0,n,d] contiguous → offset n*D + d
        in4 = tl.load(in4_ptr + n * D + d_idx, mask=d_mask, other=0.0)
        # in0[k,d] contiguous → offset k*D + d
        in0 = tl.load(in0_ptr + k * D + d_idx, mask=d_mask, other=0.0)
        # out[0,n,k,d] contiguous → offset (n*K+k)*D + d
        tl.store(out_ptr + (n * K + k) * D + d_idx, in4 - in0, mask=d_mask)


@torch.fx.wrap
def fused_broadcast_sub(in_0, in_4):
    # Pass tensors directly — no reshape/contiguous (blocked by API validator).
    # Kernel offsets are correct for contiguous tensors.
    K, D = in_0.shape
    N = in_4.shape[1]
    out = torch.empty((1, N, K, D), dtype=in_0.dtype, device=in_0.device)
    _broadcast_sub_kernel[(N, K)](in_4, in_0, out, N, K, D)
    return out


def replacement_func():
    return fused_broadcast_sub