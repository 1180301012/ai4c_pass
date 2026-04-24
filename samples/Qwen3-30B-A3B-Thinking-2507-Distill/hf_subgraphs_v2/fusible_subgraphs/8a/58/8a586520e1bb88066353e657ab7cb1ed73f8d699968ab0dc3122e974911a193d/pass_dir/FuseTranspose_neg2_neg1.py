import torch
import triton
import triton.language as tl


def pattern(in_2):
    tmp_4 = in_2.transpose(-2, -1)
    return tmp_4


def replacement_args(in_2):
    return (in_2,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['BM', 'BN'],
)
@triton.jit
def _transpose_4d_kernel(
    in_ptr,
    out_ptr,
    BM,
    BN,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bm = tl.program_id(0)
    pid_bn = tl.program_id(1)
    pid_b  = tl.program_id(2)

    rm = pid_bm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_bn * BLOCK_N + tl.arange(0, BLOCK_N)

    base_in  = pid_b * BM * BN
    base_out = pid_b * BN * BM

    rmask  = rm < BM
    rmask2 = rn < BN

    data = tl.load(
        in_ptr + base_in + rm[:, None] * BN + rn[None, :],
        mask=rmask[:, None] & rmask2[None, :],
        other=0.0,
    )

    tl.store(
        out_ptr + base_out + rn[:, None] * BM + rm[None, :],
        tl.trans(data),
        mask=rmask2[:, None] & rmask[None, :],
    )


@torch.fx.wrap
def _triton_transpose_neg2_neg1_impl(in_2):
    device = in_2.device
    dtype  = in_2.dtype
    B, C, M, N = in_2.shape
    num_batches = B * C
    in2_c = in_2.contiguous()
    out   = torch.empty((B, C, N, M), dtype=dtype, device=device)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
        num_batches,
    )
    _transpose_4d_kernel[grid](in2_c, out, M, N)
    return out


def replacement_func():
    return _triton_transpose_neg2_neg1_impl