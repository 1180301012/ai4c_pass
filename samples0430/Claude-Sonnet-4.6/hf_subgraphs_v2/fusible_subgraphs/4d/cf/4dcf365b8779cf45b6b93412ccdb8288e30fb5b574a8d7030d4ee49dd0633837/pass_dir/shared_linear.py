"""
Shared Triton kernel for fused linear (GEMM + bias).
Computes: C = A @ B.T + bias
  A: [M, K]  (input, contiguous)
  B: [N, K]  (weight, contiguous)
  bias: [N]
  C: [M, N]  (contiguous)

Design choices
--------------
* tl.make_block_ptr  — enables async hardware prefetching (cp.async on Ampere).
* GROUP_M L2 swizzle — consecutive programs share the same B tile → reduces
  effective B bandwidth; matters most for bigbird's tiny M=17.
* No BLOCK_M=128×BLOCK_N=128 config  — avoids 1-program execution on RECT_L.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # ---- bigbird: M=17, K=768, N=3072 ----
        # BLOCK_M=16 → 2 M-blocks; BLOCK_M=32 → 1 M-block
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        # ---- RECT_L: M=128, K=128, N=128 ----
        # BLOCK_K=128 → 1 K-iter (single tl.dot, no loop overhead)
        # NOTE: avoid BLOCK_M=128×BLOCK_N=128 which gives only 1 program for RECT_L
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        # BLOCK_K=64 → 2 K-iters
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        # BLOCK_K=32
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    C = A @ B.T + bias  (all matrices contiguous).
    Uses tl.make_block_ptr for async hardware prefetching.
    GROUP_M swizzle: groups consecutive M-blocks per N-block so they share
    the B tile in L2 cache → reduces effective B memory traffic.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # L2 swizzle: within each "group", iterate over M fast and N slow
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block pointer for A  ([M, K] row-major, strides K, 1)
    a_blk = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(K, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    # Block pointer for B.T  (B=[N,K], strides K,1 → B.T=[K,N], strides 1,K)
    bt_blk = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(1, K),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_blk,  boundary_check=(0, 1), padding_option="zero")
        b = tl.load(bt_blk, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b)
        a_blk  = tl.advance(a_blk,  (0, BLOCK_K))
        bt_blk = tl.advance(bt_blk, (BLOCK_K, 0))

    # Fused bias  (N-dim is always aligned → safe unmasked load)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.)
    acc += bias[None, :].to(tl.float32)

    # Store result
    c_blk = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_blk, acc.to(C_ptr.dtype.element_ty), boundary_check=(0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Specialised kernel for bigbird: K=768, N=3072 hard-coded
# Compile-time loop count (768//BLOCK_K), compile-time N-stride (3072).
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M'],  # K=768, N=3072 are constants; only M varies (batch×seq)
)
@triton.jit
def _bigbird_768_3072_kernel(A_ptr, B_ptr, Bias_ptr, C_ptr, M,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                              BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    """
    Bigbird specialised kernel: K=768 and N=3072 are Python literals → Triton
    sees them as compile-time constants, enabling:
      • Exact loop count (768 // BLOCK_K) — no tl.cdiv at runtime.
      • Strides (768, 3072) folded as immediates.
      • B.T loads and bias loads: no boundary check (K%BK=0, N%BN=0).
      • Only M boundary check remains (M=17 is not always ≥ BLOCK_M).
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = 3072 // BLOCK_N            # compile-time constant
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    # A: [M, 768]  (stride 768 is compile-time)
    a_blk = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, 768),
        strides=(768, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    # B.T: [768, 3072] (stride 1, 768 are compile-time)
    bt_blk = tl.make_block_ptr(
        base=B_ptr,
        shape=(768, 3072),
        strides=(1, 768),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(768 // BLOCK_K):        # compile-time loop count!
        a  = tl.load(a_blk,  boundary_check=(0,), padding_option="zero")  # M-only check
        b  = tl.load(bt_blk, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b)
        a_blk  = tl.advance(a_blk,  (0, BLOCK_K))
        bt_blk = tl.advance(bt_blk, (BLOCK_K, 0))

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(Bias_ptr + offs_n)      # no mask — N%BLOCK_N=0
    acc += bias[None, :].to(tl.float32)

    c_blk = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, 3072),
        strides=(3072, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_blk, acc.to(C_ptr.dtype.element_ty), boundary_check=(0,))  # M-only


# ─────────────────────────────────────────────────────────────────────────────
# Specialised kernel for RECT_L: M=K=N=128, hard-coded → no loop, no masking
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        # BLOCK_K is always 128 (= K) → single tl.dot, zero loop overhead
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
    ],
    key=[],   # M=K=N=128 is always the same → benchmark once, cache forever
)
@triton.jit
def _rectl_128_kernel(A_ptr, B_ptr, Bias_ptr, C_ptr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Specialised for M=K=N=128, BLOCK_K=128 (the entire K in one tile).
    All dimensions aligned → no masking, no K-loop.
    Hard-coded strides (128) are compile-time constants for Triton.
    """
    pid = tl.program_id(0)
    pid_m = pid // (128 // BLOCK_N)   # 128/BLOCK_N is a compile-time constant
    pid_n = pid % (128 // BLOCK_N)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 128)                         # BLOCK_K = K = 128

    # Load A[offs_m, :] and B[offs_n, :] — no masking (all aligned to 128)
    a = tl.load(A_ptr + offs_m[:, None] * 128 + offs_k[None, :])   # [BM, 128]
    b = tl.load(B_ptr + offs_n[None, :] * 128 + offs_k[:, None])   # [128, BN]
    acc = tl.dot(a, b, out_dtype=tl.float32)                        # single dot

    bias = tl.load(Bias_ptr + offs_n)
    acc = acc + bias[None, :].to(tl.float32)

    tl.store(C_ptr + offs_m[:, None] * 128 + offs_n[None, :],
             acc.to(C_ptr.dtype.element_ty))


@torch.fx.wrap
def fused_linear(bias, weight, x):
    """
    Kernel wrapper: fused dropout-identity + linear.
    Dispatches to a shape-specialised kernel when M=K=N=128 (RECT_L),
    otherwise uses the general grouped-GEMM kernel (bigbird).
    Only torch.empty is used for allocation; no .reshape/.to/.view.
    """
    K = x.shape[-1]       # Python int — no aten dispatch
    N = weight.shape[0]   # Python int
    M = x.numel() // K    # Python int

    # Allocate output in the correct final shape — no reshape needed
    out = torch.empty(x.shape[:-1] + (N,), dtype=weight.dtype, device=x.device)

    if M == 128 and N == 128 and K == 128:
        # RECT_L: M=K=N=128 — use hard-coded single-dot specialised kernel
        def grid_rectl(META):
            return (triton.cdiv(128, META['BLOCK_M']) * triton.cdiv(128, META['BLOCK_N']),)
        _rectl_128_kernel[grid_rectl](x, weight, bias, out)
    elif N == 3072 and K == 768:
        # bigbird: K=768, N=3072 hard-coded → compile-time loop + strides
        def grid_bigbird(META):
            return (triton.cdiv(M, META['BLOCK_M']) * (3072 // META['BLOCK_N']),)
        _bigbird_768_3072_kernel[grid_bigbird](x, weight, bias, out, M)
    else:
        # general fallback
        def grid_general(META):
            return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        _matmul_bias_kernel[grid_general](x, weight, bias, out, M, N, K)

    return out