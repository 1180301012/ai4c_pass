import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_I': 32,  'BLOCK_J': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_I': 64,  'BLOCK_J': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_I': 32,  'BLOCK_J': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_I': 64,  'BLOCK_J': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_I': 128, 'BLOCK_J': 32,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_I': 32,  'BLOCK_J': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_I': 128, 'BLOCK_J': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_I': 64,  'BLOCK_J': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_I': 128, 'BLOCK_J': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_I': 256, 'BLOCK_J': 32,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_I': 32,  'BLOCK_J': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_I': 256, 'BLOCK_J': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_I': 64,  'BLOCK_J': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['B', 'N', 'K', 'M'],
)
@triton.jit
def fused_scaled_softmax_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    B, N, K, M,
    stride_ab, stride_ak,
    stride_bb, stride_bk, stride_bm,
    stride_cb, stride_cm, stride_cn,
    IS_FP32: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_bi = tl.program_id(0)   # encodes rows of A / cols of C
    pid_cj = tl.program_id(1)   # encodes cols of B / rows of C

    # Decompose flat program id into (b, i_block)
    b       = pid_bi // N
    i_start = (pid_bi % N) * BLOCK_I

    i_offs = i_start + tl.arange(0, BLOCK_I)  # shape [BLOCK_I]
    j_offs = pid_cj * BLOCK_J + tl.arange(0, BLOCK_J)  # shape [BLOCK_J]

    i_mask = i_offs < N
    j_mask = j_offs < M
    k_mask = tl.arange(0, BLOCK_K) < K  # first BLOCK_K positions are valid

    # ---- Load attention logits in original dtype ----
    a_raw = tl.load(
        A_ptr + b * stride_ab + i_offs[:, None] * stride_ak + tl.arange(0, BLOCK_K)[None, :],
        mask=i_mask[:, None] & k_mask[None, :],
        other=0.0,
    )  # dtype = original (fp16/bf16/fp32), shape [BLOCK_I, BLOCK_K]

    # ---- Softmax in float32 for numerical stability ----
    a_f32 = a_raw.to(tl.float32)
    max_val = tl.max(a_f32, axis=1, keep_dims=True)          # [BLOCK_I, 1]
    a_f32   = a_f32 - max_val                                 # shift
    exp_w   = tl.exp(a_f32)                                   # [BLOCK_I, BLOCK_K]
    sum_w   = tl.sum(exp_w, axis=1, keep_dims=True)           # [BLOCK_I, 1]
    attn_w  = exp_w / sum_w                                    # [BLOCK_I, BLOCK_K], float32

    # Convert back to original dtype for the matmul
    attn = attn_w.to(a_raw.dtype)  # [BLOCK_I, BLOCK_K], original dtype

    # ---- Load B tile [BLOCK_K, BLOCK_J] ----
    b_tile = tl.load(
        B_ptr + b * stride_bb
                 + tl.arange(0, BLOCK_K)[:, None] * stride_bk
                 + j_offs[None, :] * stride_bm,
        mask=k_mask[:, None] & j_mask[None, :],
        other=0.0,
    )  # shape [BLOCK_K, BLOCK_J], dtype = original

    # ---- Fused matmul: C_tile = attn @ b_tile ----
    # Accumulate in float32; tl.dot uses fp32 accumulator for fp16/bf16 inputs
    acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
    if IS_FP32:
        acc = tl.dot(attn.to(tl.float32), b_tile.to(tl.float32), acc)
    else:
        acc = tl.dot(attn, b_tile, acc)

    # ---- Store C at transposed position: C[b, j, i] ----
    tl.store(
        C_ptr + b * stride_cb
               + j_offs[None, :] * stride_cn
               + i_offs[:, None] * stride_cm,
        acc,
        mask=j_mask[None, :] & i_mask[:, None],
    )


@torch.fx.wrap
def fused_scaled_softmax_matmul(in_0, in_1):
    """
    Fused kernel replacing:
        tmp_0  = 0.0625 * in_0
        tmp_1  = softmax(tmp_0, dim=-1)
        matmul = tmp_1 @ in_1
        out    = matmul.permute(0, 2, 1)
    Inputs:
        in_0 : [B, N, K]     (e.g. [B, 8192, 19])
        in_1 : [B, K, M]     (e.g. [B, 19, 256])
    Output:
        out  : [B, M, N]     (e.g. [B, 256, 8192])
    """
    B, N, K = in_0.shape
    _B, _K, M = in_1.shape
    assert _B == B and _K == K

    IS_FP32 = in_0.dtype == torch.float32
    IS_BF16 = in_0.dtype == torch.bfloat16
    IS_FP16 = not IS_FP32 and not IS_BF16
    output_dtype = in_0.dtype

    out = torch.empty((B, M, N), dtype=output_dtype, device=in_0.device)

    grid = lambda meta: (
        B * N,
        triton.cdiv(M, meta['BLOCK_J']),
    )

    fused_scaled_softmax_matmul_kernel[grid](
        in_0, in_1, out,
        B, N, K, M,
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        out.stride(0),  out.stride(1),  out.stride(2),
        IS_FP32=IS_FP32,
        IS_BF16=IS_BF16,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement hooks required by the AI4C framework
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    """Mirrors model.py exactly."""
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    matmul = torch.matmul(tmp_1, in_1)
    tmp_3  = matmul.permute(0, 2, 1)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_scaled_softmax_matmul