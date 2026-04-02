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
        triton.Config({'BLOCK_D': 512}, num_warps=4),
        triton.Config({'BLOCK_D': 512}, num_warps=8),
        triton.Config({'BLOCK_D': 256}, num_warps=4),
        triton.Config({'BLOCK_D': 256}, num_warps=8),
        triton.Config({'BLOCK_D': 128}, num_warps=4),
    ],
    key=['N', 'K', 'D'],
)
@triton.jit
def broadcast_sub_kernel(
    in4_ptr, in0_ptr, out_ptr,
    N, K, D,
    BLOCK_D: tl.constexpr,
):
    """
    Each program handles one (n, k) pair.
    out[0, n, k, :] = in4[0, n, :] - in0[k, :]
    
    in4 layout: [1, N, D]   -> in4[0,n,d]   = n*D + d
    in0 layout: [K, D]      -> in0[k,d]     = k*D + d
    out layout: [1, N, K, D] -> out[0,n,k,d] = n*K*D + k*D + d
    """
    n = tl.program_id(0)  # 0..N-1
    k = tl.program_id(1)  # 0..K-1

    # Process D elements in chunks of BLOCK_D
    num_chunks = D // BLOCK_D
    for chunk in range(num_chunks):
        d_start = chunk * BLOCK_D
        d_range = d_start + tl.arange(0, BLOCK_D)   # [BLOCK_D]

        in4_data = tl.load(in4_ptr + n * D + d_range)   # in4[0,n,d_range]
        in0_data = tl.load(in0_ptr + k * D + d_range)   # in0[k, d_range]

        diff = in4_data - in0_data

        out_offset = n * K * D + k * D + d_range        # out[0,n,k,d_range]
        tl.store(out_ptr + out_offset, diff)


@torch.fx.wrap
def fused_broadcast_subtract(in_0, in_4):
    """
    in_0: [K,  D]      = [32,   512]  - codewords
    in_4: [1, N, D]    = [1, 4096, 512] - feature maps
    Returns: [1, N, K, D] = [1, 4096, 32, 512]
             out[0,n,k,d] = in_4[0,n,d] - in_0[k,d]
    """
    K, D = in_0.shape          # 32, 512
    N   = in_4.shape[1]        # 4096

    out = torch.empty((1, N, K, D), dtype=in_0.dtype, device=in_0.device)

    in_0_c = in_0.contiguous()
    in_4_c = in_4.contiguous()

    grid = (N, K)              # (4096, 32)
    broadcast_sub_kernel[grid](
        in_4_c, in_0_c, out,
        N, K, D,
    )
    return out


def replacement_func():
    return fused_broadcast_subtract