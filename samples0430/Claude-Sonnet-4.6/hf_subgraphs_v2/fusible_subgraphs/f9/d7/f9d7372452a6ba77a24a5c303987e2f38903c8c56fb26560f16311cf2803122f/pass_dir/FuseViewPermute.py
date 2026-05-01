import torch
import triton
import triton.language as tl


def pattern(in_1):
    tmp_3 = in_1.view(1, 32, -1)
    tmp_4 = tmp_3.permute(0, 2, 1)
    return tmp_4


def replacement_args(in_1):
    return (in_1,)


@triton.jit
def _view_permute_kernel(
    in_ptr,
    out_ptr,
    BLOCK_N: tl.constexpr,
):
    # Transpose [32, 3072] -> [3072, 32]  (batch=1 ignored)
    # in[m, n] at offset m*3072 + n  (M=32 fixed, N=3072 fixed)
    # out[n, m] at offset n*32 + m
    pid = tl.program_id(0)

    m_offs = tl.arange(0, 32)                          # [32]
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)     # [BLOCK_N]

    # Load tile [32, BLOCK_N]  – coalesced row reads
    in_offs = m_offs[:, None] * 3072 + n_offs[None, :]  # [32, BLOCK_N]
    data = tl.load(in_ptr + in_offs)

    # Store transposed tile [BLOCK_N, 32]
    out_offs = n_offs[:, None] * 32 + m_offs[None, :]   # [BLOCK_N, 32]
    tl.store(out_ptr + out_offs, tl.trans(data))


@torch.fx.wrap
def fused_view_permute(in_1):
    # in_1: [1, 32, 64, 48]  → view [1, 32, 3072] → permute(0,2,1) [1, 3072, 32]
    # 3072 = 64*48; 3072 divisible by 64, so no masking needed
    BLOCK_N = 64          # 48 programs
    out = torch.empty(1, 3072, 32, dtype=in_1.dtype, device=in_1.device)
    _view_permute_kernel[(48,)](in_1, out, BLOCK_N=BLOCK_N)
    return out


def replacement_func():
    return fused_view_permute