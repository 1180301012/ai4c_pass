import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: 1×1 Conv2d (stride=1, pad=0, dil=1, groups=1) followed by flatten
# from dim 2.  This is a pure batched GEMM:
#   out[n, o, m] = bias[o] + sum_k( inp[n, k, m] * w[o, k] )
# where m = h*W + w  (spatial positions flattened).
# ──────────────────────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(conv2d, 2)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    # in_0 = bias  [C_out]
    # in_1 = weight [C_out, C_in, 1, 1]
    # in_2 = input  [N, C_in, H, W]
    return (in_0, in_1, in_2)


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel
#   Grid: (ceil(M/BM), ceil(C_out/BN), N)
#   Each program computes a [BN, BM] tile of the output matrix for one batch.
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        # 4 configs with BLOCK_M ≥ 128.
        # For N=128: worst grid = 24*2*128 = 6144 programs (BM=128,BN=16) vs
        #            best  grid = 12*1*128 = 1536 programs (BM=256,BN=32).
        # No config can produce catastrophically large grids.
        # With 4 configs × ~6 warmup runs each in 25 iterations → reliable.
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['N_batch', 'M', 'N_OUT', 'K'],
)
@triton.jit
def fused_conv1x1_flatten_kernel(
    inp_ptr,   # [N, K, M]  (viewed as contiguous from [N, K, H, W])
    w_ptr,     # [N_OUT, K] (viewed as contiguous from [N_OUT, K, 1, 1])
    bias_ptr,  # [N_OUT]
    out_ptr,   # [N, N_OUT, M]
    N_batch,
    M,
    N_OUT,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m    = tl.program_id(0)   # tile over spatial dim M
    pid_n    = tl.program_id(1)   # tile over output-channel dim N_OUT
    batch_id = tl.program_id(2)   # one program per batch element

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # ── Block pointer for input: logical view [K, M] per batch item
    # Strides (M, 1): row k starts at inp_ptr[batch*K*M + k*M], cols are contiguous
    inp_block = tl.make_block_ptr(
        base=inp_ptr + batch_id * K * M,
        shape=(K, M),
        strides=(M, 1),
        offsets=(0, m_start),
        block_shape=(BLOCK_K, BLOCK_M),
        order=(1, 0),   # M-dim (last) varies fastest → row-major reads
    )

    # ── Block pointer for weight: logical view [N_OUT, K]
    # Strides (K, 1): weight stored as [N_OUT, K, 1, 1] in memory (contiguous)
    w_block = tl.make_block_ptr(
        base=w_ptr,
        shape=(N_OUT, K),
        strides=(K, 1),
        offsets=(n_start, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )

    # ── Accumulate in float32 for precision
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        # Block-ptr loads enable automatic software pipelining (num_stages)
        # boundary_check on dim-0 handles N_OUT=17 not divisible by BLOCK_N.
        # K and M are always divisible by BLOCK_K/BLOCK_M (no check needed there).
        inp = tl.load(inp_block)                                          # [BLOCK_K, BLOCK_M]
        w   = tl.load(w_block, boundary_check=(0,), padding_option="zero")  # [BLOCK_N, BLOCK_K]

        # Tensor-core GEMM: [BLOCK_N, BLOCK_K] @ [BLOCK_K, BLOCK_M] → [BLOCK_N, BLOCK_M]
        acc = tl.dot(w, inp, acc)

        # Advance both block pointers along K
        inp_block = tl.advance(inp_block, (BLOCK_K, 0))
        w_block   = tl.advance(w_block,   (0, BLOCK_K))

    # ── Add bias  [BLOCK_N]
    n_offs = n_start + tl.arange(0, BLOCK_N)
    m_offs = m_start + tl.arange(0, BLOCK_M)
    bias   = tl.load(bias_ptr + n_offs, mask=n_offs < N_OUT, other=0.0)
    acc    = acc + bias[:, None]

    # ── Store with explicit pointers (supports implicit fp32→bf16/fp16 cast;
    #    block-ptr tl.store requires value dtype to match pointer element type)
    out_ptrs = out_ptr + batch_id * N_OUT * M + n_offs[:, None] * M + m_offs[None, :]
    out_mask = (n_offs[:, None] < N_OUT) & (m_offs[None, :] < M)
    tl.store(out_ptrs, acc, mask=out_mask)


# ──────────────────────────────────────────────────────────────────────────────
# Python wrapper  (must be decorated with @torch.fx.wrap)
# ──────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_conv1x1_flatten(bias, weight, inp):
    """
    bias   : [C_out]                  e.g. [17]
    weight : [C_out, C_in, 1, 1]      e.g. [17, 160, 1, 1]
    inp    : [N, C_in, H, W]           e.g. [N, 160, 64, 48]
    returns: [N, C_out, H*W]           e.g. [N, 17, 3072]
    """
    N    = inp.shape[0]
    C_in = inp.shape[1]
    H    = inp.shape[2]
    W    = inp.shape[3]
    C_out = weight.shape[0]
    M     = H * W                          # spatial positions = 3072

    out = torch.empty((N, C_out, M), dtype=inp.dtype, device=inp.device)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
        N,
    )

    fused_conv1x1_flatten_kernel[grid](
        inp, weight, bias, out,
        N, M, C_out, C_in,
    )

    return out


def replacement_func():
    return fused_conv1x1_flatten