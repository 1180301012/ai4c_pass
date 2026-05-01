import torch
import triton
import triton.language as tl


@triton.jit
def attn_mask_kernel(
    in0_ptr,  # [batch, N] int64 - attention mask
    in2_ptr,  # [N] int64 - cache positions
    out_ptr,  # [batch, 1, N, N] bool (stored as int8)
    batch,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused attention mask kernel.
    Computes: out[b,0,i,j] = (j <= in2[i]) AND bool(in0[b,j])
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = batch * N * N
    mask = offs < total

    b = offs // (N * N)
    ij = offs % (N * N)
    i = ij // N   # row
    j = ij % N    # col

    # Load in2[i] (causal/cache position) as int64
    in2_val = tl.load(in2_ptr + i, mask=mask, other=0).to(tl.int64)

    # Load in0[b, j] (attention mask value) as int64
    in0_val = tl.load(in0_ptr + b * N + j, mask=mask, other=0)

    # result[b,0,i,j] = (j <= in2[i]) AND bool(in0[b,j])
    j_i64 = j.to(tl.int64)
    causal = j_i64 <= in2_val
    attn = in0_val != 0
    result = causal & attn

    tl.store(out_ptr + offs, result.to(tl.int8), mask=mask)


@triton.jit
def cast_to_float32_kernel(
    in_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Cast any dtype to float32, element-wise (contiguous memory)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    val = tl.load(in_ptr + offs, mask=mask).to(tl.float32)
    tl.store(out_ptr + offs, val, mask=mask)