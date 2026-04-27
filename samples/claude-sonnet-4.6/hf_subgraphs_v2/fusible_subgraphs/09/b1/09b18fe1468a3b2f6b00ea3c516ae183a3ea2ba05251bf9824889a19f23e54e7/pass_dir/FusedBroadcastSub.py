import torch
import triton
import triton.language as tl


def pattern(in_0, in_4):
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    tmp_10 = tmp_8 - tmp_6
    return tmp_10


def replacement_args(in_0, in_4):
    return (in_0, in_4)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_D': 256}, num_warps=4),
        triton.Config({'BLOCK_D': 256}, num_warps=8),
        triton.Config({'BLOCK_D': 512}, num_warps=4),
        triton.Config({'BLOCK_D': 512}, num_warps=8),
    ],
    key=['D'],
)
@triton.jit
def _broadcast_sub_kernel(
    in4_ptr,    # [N, D]
    in0_ptr,    # [K, D]
    out_ptr,    # [N, K, D]
    N, K, D,
    BLOCK_D: tl.constexpr,
):
    # 2D grid: dim0 = n, dim1 = k
    n = tl.program_id(0)
    k = tl.program_id(1)

    for d_start in range(0, D, BLOCK_D):
        d_idx = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_idx < D

        # Load in4[n, d] - broadcasted across all k
        in4 = tl.load(in4_ptr + n * D + d_idx, mask=d_mask, other=0.0)
        # Load in0[k, d] - broadcasted across all n
        in0 = tl.load(in0_ptr + k * D + d_idx, mask=d_mask, other=0.0)

        out = in4 - in0

        tl.store(out_ptr + (n * K + k) * D + d_idx, out, mask=d_mask)


@torch.fx.wrap
def fused_broadcast_sub(in_0, in_4):
    # in_0: [32, 512]  (codewords: K x D)
    # in_4: [1, 4096, 512]  (features: 1 x N x D)
    K, D = in_0.shape
    N = in_4.shape[1]

    in4_flat = in_4.contiguous().reshape(N, D)
    in0_c = in_0.contiguous()

    # Output: [1, N, K, D]
    out = torch.empty((1, N, K, D), dtype=in_0.dtype, device=in_0.device)
    out_flat = out.reshape(N, K, D)

    # 2D grid: (N, K) = (4096, 32)
    _broadcast_sub_kernel[(N, K)](
        in4_flat, in0_c, out_flat,
        N, K, D,
    )

    return out


def replacement_func():
    return fused_broadcast_sub