import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, a, b):
    tmp_0 = torch.cat([in_1, in_0])
    tmp_1 = torch.arange(24)
    tmp_2 = torch.arange(24)
    meshgrid = torch.functional.meshgrid(tmp_1, tmp_2, indexing='ij')
    tmp_4 = meshgrid[0]
    tmp_5 = meshgrid[1]
    tmp_6 = torch.stack((tmp_4, tmp_5))
    tmp_7 = torch.flatten(tmp_6, 1)
    tmp_8 = tmp_7[(slice(None, None, None), slice(None, None, None), None)]
    tmp_9 = tmp_7[(slice(None, None, None), None, slice(None, None, None))]
    tmp_10 = tmp_8 - tmp_9
    tmp_11 = tmp_10.permute(1, 2, 0)
    tmp_12 = tmp_11.contiguous()
    tmp_13 = tmp_12[(slice(None, None, None), slice(None, None, None), 0)]
    tmp_13 += 23
    tmp_12[(slice(None, None, None), slice(None, None, None), 0)] = tmp_13
    tmp_16 = tmp_12[(slice(None, None, None), slice(None, None, None), 1)]
    tmp_16 += 23
    tmp_12[(slice(None, None, None), slice(None, None, None), 1)] = tmp_16
    tmp_19 = tmp_12[(slice(None, None, None), slice(None, None, None), 0)]
    tmp_19 *= 47
    tmp_12[(slice(None, None, None), slice(None, None, None), 0)] = tmp_19
    tmp_22 = torch.zeros(size=(577, 577), dtype=torch.int64)
    tmp_23 = tmp_12.sum(-1)
    tmp_22[(slice(1, None, None), slice(1, None, None))] = tmp_23
    tmp_22[(0, slice(0, None, None))] = 2209
    tmp_22[(slice(0, None, None), 0)] = 2210
    tmp_22[(0, 0)] = 2211
    tmp_28 = tmp_22.view(-1)
    return (tmp_0, tmp_28)


def replacement_args(in_0, in_1, a, b):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['N', 'SQR'],
)
@triton.jit
def _kernelEdgeWeight_S47_577(out_ptr, N, SCALE, OFFSET, SQR, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    N2 = N * N
    hw = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = hw < N2

    ar = hw // N
    r = hw % N

    clue = N - r
    bc = r + OFFSET + 1
    bw = tl.maximum(N - bc, 0)
    hclamped = tl.maximum(N - 1 - ar, 0)

    edge_weight = tl.maximum(1,
        hclamped * SCALE + clue * 27 + OFFSET * SCALE + OFFSET * OFFSET)

    ow = r + OFFSET
    out_idx = ow * SQR + ow
    tl.store(out_ptr + out_idx, edge_weight, mask=mask)


@torch.fx.wrap
def _replace_FuseBatchPred_N24_S47_O23(a, b):
    N2 = 24 * 24
    SQR = 577
    SCALE = 47
    OFFSET = 23

    outFlat = torch.empty(SQR * SQR, dtype=torch.int64)

    grid = lambda meta: ((N2 + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _kernelEdgeWeight_S47_577[grid](
        outFlat, 24, SCALE, OFFSET, SQR
    )
    tmp_28 = outFlat.view(SQR, SQR)
    return (a, tmp_28)


def replacement_func():
    return _replace_FuseBatchPred_N24_S47_O23