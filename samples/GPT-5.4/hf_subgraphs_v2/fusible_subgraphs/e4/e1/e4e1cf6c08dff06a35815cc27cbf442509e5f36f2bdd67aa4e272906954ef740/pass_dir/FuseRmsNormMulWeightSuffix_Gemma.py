import torch
import triton
import triton.language as tl
from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


HIDDEN_SIZE = 2048
EPS = 1e-6
CHUNK = 512
INV_HIDDEN_SIZE = 1.0 / HIDDEN_SIZE


def pattern(tmp_2, in_1):
    tmp_4 = tmp_2.float()
    tmp_5 = tmp_4.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_9 = tmp_4 * tmp_8
    tmp_10 = in_1.float()
    tmp_11 = 1.0 + tmp_10
    tmp_12 = tmp_9 * tmp_11
    tmp_13 = tmp_12.type_as(tmp_2)
    return tmp_13


def replacement_args(tmp_2, in_1):
    return (tmp_2, in_1)


@triton.jit
def _rmsnorm_2048_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    stride_row,
    CHUNK_SIZE: tl.constexpr,
    INV_HIDDEN_SIZE_CONST: tl.constexpr,
    EPS_CONST: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * stride_row
    offs = tl.arange(0, CHUNK_SIZE)

    x0 = tl.load(x_ptr + row_start + offs)
    x1 = tl.load(x_ptr + row_start + CHUNK_SIZE + offs)
    x2 = tl.load(x_ptr + row_start + 2 * CHUNK_SIZE + offs)
    x3 = tl.load(x_ptr + row_start + 3 * CHUNK_SIZE + offs)

    x0f = x0.to(tl.float32)
    x1f = x1.to(tl.float32)
    x2f = x2.to(tl.float32)
    x3f = x3.to(tl.float32)

    sumsq = tl.sum(x0f * x0f, axis=0)
    sumsq += tl.sum(x1f * x1f, axis=0)
    sumsq += tl.sum(x2f * x2f, axis=0)
    sumsq += tl.sum(x3f * x3f, axis=0)
    inv_rms = tl.rsqrt(sumsq * INV_HIDDEN_SIZE_CONST + EPS_CONST)

    w0 = tl.load(w_ptr + offs).to(tl.float32)
    w1 = tl.load(w_ptr + CHUNK_SIZE + offs).to(tl.float32)
    w2 = tl.load(w_ptr + 2 * CHUNK_SIZE + offs).to(tl.float32)
    w3 = tl.load(w_ptr + 3 * CHUNK_SIZE + offs).to(tl.float32)

    y0 = x0f * inv_rms * (1.0 + w0)
    y1 = x1f * inv_rms * (1.0 + w1)
    y2 = x2f * inv_rms * (1.0 + w2)
    y3 = x3f * inv_rms * (1.0 + w3)

    tl.store(out_ptr + row_start + offs, y0.to(tl.bfloat16))
    tl.store(out_ptr + row_start + CHUNK_SIZE + offs, y1.to(tl.bfloat16))
    tl.store(out_ptr + row_start + 2 * CHUNK_SIZE + offs, y2.to(tl.bfloat16))
    tl.store(out_ptr + row_start + 3 * CHUNK_SIZE + offs, y3.to(tl.bfloat16))


@triton.jit
def _rmsnorm_generic_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    H,
    EPS_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * H

    sumsq = 0.0
    for start in tl.static_range(0, H, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0).to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)

    inv_rms = tl.rsqrt(sumsq / H + EPS_CONST)

    for start in tl.static_range(0, H, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0).to(tl.float32)
        y = x * inv_rms * (1.0 + w)
        tl.store(out_ptr + row_start + cols, y.to(tl.bfloat16), mask=mask)


@torch.fx.wrap
def fused_rmsnorm_mul_weight(tmp_2, in_1):
    tmp_2 = unwrap_tensor(tmp_2)
    in_1 = unwrap_tensor(in_1)
    hidden_size = in_1.numel()
    n_rows = tmp_2.numel() // hidden_size
    out = torch.empty_like(tmp_2)
    if hidden_size == HIDDEN_SIZE:
        _rmsnorm_2048_kernel[(n_rows,)](
            tmp_2,
            in_1,
            out,
            stride_row=hidden_size,
            CHUNK_SIZE=CHUNK,
            INV_HIDDEN_SIZE_CONST=INV_HIDDEN_SIZE,
            EPS_CONST=EPS,
            num_warps=4,
            num_stages=1,
        )
    else:
        _rmsnorm_generic_kernel[(n_rows,)](
            tmp_2,
            in_1,
            out,
            H=hidden_size,
            EPS_CONST=EPS,
            BLOCK_SIZE=512,
            num_warps=4,
            num_stages=1,
        )
    return out


def replacement_func():
    return fused_rmsnorm_mul_weight