import torch
import triton
import triton.language as tl


@triton.jit
def attn_mask_bool_kernel(
    in_0_ptr,   # [B, N] int64 attention mask (non-zero = True)
    in_2_ptr,   # [N] int64 position indices
    out_ptr,    # [B, 1, 1, N] bool output
    B,
    N,
    BLOCK_N: tl.constexpr,
):
    """
    Each program handles row (b, k): output[b, 0, 0, j] = bool(in_0[b,j]) AND (k >= j).
    Grid = (B * N,)
    """
    pid = tl.program_id(0)
    b = pid // N
    k = pid % N

    j = tl.arange(0, BLOCK_N)
    mask = j < N

    # Load in_0[b, j]  → int64; non-zero → True
    attn = tl.load(in_0_ptr + b * N + j, mask=mask, other=0)
    attn_bool = attn != 0

    # Causal: k >= j
    causal = k >= j

    result = attn_bool & causal

    # out[b, 0, 0, j] lives at offset b*N + j  (shape [B,1,1,N], contiguous)
    tl.store(out_ptr + b * N + j, result, mask=mask)


@triton.jit
def rotary_expand_kernel(
    src_ptr,    # [M] input (any dtype)
    out_ptr,    # [1, 1, M] float32 output
    M,
    BLOCK_M: tl.constexpr,
):
    """
    Each program handles one element k.
    out[0, 0, k] = float(src[k]).
    Grid = (M,)
    """
    pid = tl.program_id(0)
    k = pid

    val = tl.load(src_ptr + k)
    result = val.to(tl.float32)

    # out[0, 0, k] at offset k
    tl.store(out_ptr + k, result)