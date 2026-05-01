"""
Shared Triton kernel for 1x1 convolution (NCHW in, NCHW out).

Strategy
--------
Stride-1:
  • Single-kernel path: conv1x1_s1_kernel reads NCHW input with incremental
    pointer updates (pre-computed base + one add per K-loop iter).
  • Tile layout [BLOCK_K, BLOCK_M] (K outer, M inner) gives stride-1 reads
    within each K-row (consecutive spatial positions in the same batch).

Stride-2:
  • 2-step path: extract stride-2 pixels into a contiguous [N,Cin,Ho,Wo]
    buffer (one pass at 50% coalescing), then run the same stride-1 GEMM
    on the contiguous buffer (fully coalesced reads).
  • This avoids re-reading the non-coalesced stride-2 data N_out/BLOCK_N
    (~18) times inside the GEMM loop.

Both passes share the same replacement_func (dispatch_conv1x1) distinguished
by a route string "s1" / "s2" in replacement_args.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Stride-2 pixel extraction kernel
# ---------------------------------------------------------------------------
@triton.jit
def extract_s2_kernel(
    input_ptr, output_ptr,
    N_batch, Cin, H, W, H_out, W_out, HW_out, HW_pad,
    BLOCK_C: tl.constexpr,   # channels per tile
    BLOCK_M: tl.constexpr,   # spatial positions per tile  (keep ≤ H_out*W_out)
):
    """
    output[n, c, hw] = input[n, c, 2*(hw//W_out), 2*(hw%W_out)]
    Output layout: [N_batch, Cin, HW_pad]  (HW_pad ≥ HW_out, padded to multiple of 64)
    Grid: (cdiv(N_batch*HW_out, BLOCK_M), cdiv(Cin, BLOCK_C))

    Tile layout [BLOCK_C, BLOCK_M] (C outer, M inner):
      • Writes (output): fast M dim → stride-1 per batch → COALESCED ✓
      • Reads  (input) : fast M dim → stride-2 in W     → ~50% coalesced
    """
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    M_total = N_batch * HW_out

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)  # [BLOCK_C]

    m_mask = m_offs < M_total
    c_mask = c_offs < Cin

    # Decode m → (batch, h_out, w_out)
    n       = m_offs // HW_out        # [BLOCK_M]
    hw      = m_offs % HW_out         # [BLOCK_M]
    h_out_v = hw // W_out             # [BLOCK_M]
    w_out_v = hw % W_out              # [BLOCK_M]

    # Input base (batch + strided spatial, no C)
    a_base = n * (Cin * H * W) + (2 * h_out_v) * W + (2 * w_out_v)  # [BLOCK_M]

    # Load [BLOCK_C, BLOCK_M] from input  (C outer, M inner → coalesced in M ✓)
    a_ptrs = input_ptr + c_offs[:, None] * (H * W) + a_base[None, :]
    val = tl.load(a_ptrs, mask=c_mask[:, None] & m_mask[None, :], other=0.0)

    # Output base using HW_PAD strides: output[n, c, hw] at n*Cin*HW_pad + c*HW_pad + hw
    o_base = n * (Cin * HW_pad) + hw   # [BLOCK_M]  (hw < HW_out ≤ HW_pad always)

    # Store [BLOCK_C, BLOCK_M] to output  (stride-1 in M within same batch ✓)
    o_ptrs = output_ptr + c_offs[:, None] * HW_pad + o_base[None, :]
    tl.store(o_ptrs, val, mask=c_mask[:, None] & m_mask[None, :])


# ---------------------------------------------------------------------------
# Batched GEMM kernel for stride-2 (3D grid: batch × M-tile × N-tile)
# Avoids batch-boundary address jumps by processing each batch independently.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=2, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_warps=8, num_stages=3),
    ],
    key=['HW', 'N', 'K'],
)
@triton.jit
def conv1x1_batched_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    Cin, HW, HW_out, Cout,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Batched GEMM: A[N_batch,Cin,HW] @ B[Cout,Cin]^T → C[N_batch,Cout,HW_out]
    HW     = HW_pad (cache-aligned, ≥ HW_out) — used for A-read strides.
    HW_out = actual output spatial size — used for C-write strides and output mask.
    """
    batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offs < HW_out   # only valid (non-padded) positions
    n_mask = n_offs < Cout

    # A reads: use HW (= HW_pad) for strides → cache-line aligned ✓
    a_base = batch * (Cin * HW) + m_offs               # [BLOCK_M]

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for k_start in range(0, Cin, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < Cin

        a_addr = a_base[None, :] + k_offs[:, None] * HW   # stride = HW (padded)
        a = tl.load(A_ptr + a_addr,
                    mask=k_mask[:, None] & m_mask[None, :], other=0.0)

        b_addr = n_offs[:, None] * Cin + k_offs[None, :]
        b = tl.load(B_ptr + b_addr,
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        acc += tl.dot(b, a)

    # C writes: use HW_out (actual) strides → correct NCHW layout
    c_ptrs = (C_ptr
              + batch * (Cout * HW_out)
              + n_offs[:, None] * HW_out
              + m_offs[None, :])
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty),
             mask=n_mask[:, None] & m_mask[None, :])



@triton.autotune(
    configs=[
        # Large tiles — good for large M, N, K (BF16/FP16 safe)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8},
                      num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8},
                      num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8},
                      num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8},
                      num_warps=4,  num_stages=4),
        # Medium tiles with BLOCK_K=32 — safe for all dtypes
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8},
                      num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8},
                      num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8},
                      num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8},
                      num_warps=4,  num_stages=5),
        # Small tiles — for small M, high parallelism
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8},
                      num_warps=4,  num_stages=5),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32,  'GROUP_M': 8},
                      num_warps=4,  num_stages=5),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32,  'GROUP_M': 8},
                      num_warps=4,  num_stages=5),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 16,  'GROUP_M': 8},
                      num_warps=2,  num_stages=5),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16,  'BLOCK_K': 16,  'GROUP_M': 8},
                      num_warps=2,  num_stages=5),
        # Extra configs for special shapes
        # BLOCK_M=16 for small M (e.g., M=1568 extracted, 32×7×7): fewer batch-boundary crossings per tile
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8},
                      num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8},
                      num_warps=2,  num_stages=5),
        # BLOCK_N=32 for small N (e.g., N=160 dpn68), better tile utilization
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32,  'GROUP_M': 8},
                      num_warps=4,  num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_s1_kernel(
    input_ptr, weight_ptr, output_ptr,
    Cin, H, W, Cout,
    M,     # N_batch * H * W
    N,     # Cout
    K,     # Cin
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    output[n, c_out, h, w] = sum_k input[n, k, h, w] * weight[c_out, k]

    Input  : [N_batch, Cin, H, W]   (stride-1 spatial — either original or extracted)
    Weight : [Cout, Cin]
    Output : [N_batch, Cout, H, W]  (NCHW)

    Tile layout:
      A[k_i, m_j]  loaded as [BLOCK_K, BLOCK_M] — coalesced in M (stride-1 spatial) ✓
      B[n_i, k_j]  loaded as [BLOCK_N, BLOCK_K] — coalesced in K (row-major weight) ✓
      C_T = B @ A  = [BLOCK_N, BLOCK_M]  written to NCHW output
      (output[n_i, m_j] stored with N_i outer, M_j inner → stride-1 per batch ✓)

    Pointer updates use incremental adds (pre-computed base, no per-loop recompute).
    """
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id     = pid // num_pid_in_group
    first_pid_m  = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    m_mask = m_offs < M
    n_mask = n_offs < N

    # --- Decode m → (batch, h, w) once per thread-block ---
    HW   = H * W
    batch = m_offs // HW        # [BLOCK_M]
    hw    = m_offs % HW         # [BLOCK_M]
    h_v   = hw // W             # [BLOCK_M]
    w_v   = hw % W              # [BLOCK_M]

    # --- Pre-compute constant base (batch + spatial, no K/C term) ---
    # Kept in registers as a small [BLOCK_M] vector throughout the K-loop.
    a_base = batch * (Cin * HW) + h_v * W + w_v          # [BLOCK_M]

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)         # [BLOCK_K]
        k_mask = k_offs < K

        # Recompute address tensors inside the loop so they are temporaries
        # (not kept in registers between iterations → lower register pressure).
        a_addr = a_base[None, :] + k_offs[:, None] * HW   # [BLOCK_K, BLOCK_M]
        a = tl.load(input_ptr + a_addr,
                    mask=k_mask[:, None] & m_mask[None, :], other=0.0)

        b_addr = n_offs[:, None] * Cin + k_offs[None, :]   # [BLOCK_N, BLOCK_K]
        b = tl.load(weight_ptr + b_addr,
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        acc += tl.dot(b, a)          # [BLOCK_N, BLOCK_K] @ [BLOCK_K, BLOCK_M] = [BLOCK_N, BLOCK_M]

    # --- Write C_T to NCHW output ---
    # c_ptrs[n_i, m_j]: for fixed n_i, varying m_j within same batch → stride-1 ✓
    c_ptrs = (output_ptr
              + batch[None, :] * (N * HW)   # [1, BLOCK_M]
              + n_offs[:, None] * HW         # [BLOCK_N, 1]
              + h_v[None, :] * W             # [1, BLOCK_M]
              + w_v[None, :])                # [1, BLOCK_M]
    # c_ptrs: [BLOCK_N, BLOCK_M]

    out_mask = n_mask[:, None] & m_mask[None, :]
    tl.store(c_ptrs, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


# ---------------------------------------------------------------------------
# Shared dispatch wrapper  (returned by BOTH Conv1x1Stride1 and Conv1x1Stride2)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def dispatch_conv1x1(weight, inp, route):
    """
    Unified entry-point.
    route = "s1" → stride-1 direct path
    route = "s2" → stride-2: extract contiguous buffer, then stride-1 GEMM
    """
    N_batch = inp.shape[0]
    Cin     = inp.shape[1]
    H       = inp.shape[2]
    W       = inp.shape[3]
    Cout    = weight.shape[0]

    # Move weight to GPU if it arrived on CPU
    w = weight if weight.device == inp.device else weight.to(inp.device)

    if route == "s1":
        # ---- Direct stride-1 path (flat GEMM with NCHW index decoding) ----
        M = N_batch * H * W

        output = torch.empty((N_batch, Cout, H, W), dtype=inp.dtype, device=inp.device)
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(Cout, meta['BLOCK_N']),
        )
        conv1x1_s1_kernel[grid](
            inp, w, output,
            Cin, H, W, Cout,
            M, Cout, Cin,
        )
        return output

    else:
        # ---- Stride-2 path: extract → batched GEMM (with HW padding for cache alignment) ----
        H_out = H // 2
        W_out = W // 2
        HW_out = H_out * W_out
        # Pad HW to next multiple of 64 for cache-line-aligned A reads (100% efficiency)
        # For HW_out=49 (7×7): HW_pad=64 → each K-row exactly 1 cache line ✓
        HW_pad = ((HW_out + 63) // 64) * 64
        if HW_pad == 0:
            HW_pad = 64

        # Step 1: extract stride-2 pixels into padded buffer [N_batch, Cin, HW_pad]
        extracted = torch.empty((N_batch, Cin, HW_pad),
                                dtype=inp.dtype, device=inp.device)
        EX_BLOCK_C = 32
        EX_BLOCK_M = 32
        ex_grid = (
            triton.cdiv(N_batch * HW_out, EX_BLOCK_M),
            triton.cdiv(Cin, EX_BLOCK_C),
        )
        extract_s2_kernel[ex_grid](
            inp, extracted,
            N_batch, Cin, H, W, H_out, W_out, HW_out, HW_pad,
            BLOCK_C=EX_BLOCK_C,
            BLOCK_M=EX_BLOCK_M,
        )

        # Step 2: batched GEMM — A uses HW_pad strides, C uses HW_out strides
        # Grid: (N_batch, cdiv(HW_pad, BLOCK_M), cdiv(Cout, BLOCK_N))
        # For HW_pad=64, BLOCK_M=64: 1 M-tile per batch → 100% A cache efficiency ✓
        output = torch.empty((N_batch, Cout, H_out, W_out),
                             dtype=inp.dtype, device=inp.device)
        grid_batched = lambda meta: (
            N_batch,
            triton.cdiv(HW_pad, meta['BLOCK_M']),
            triton.cdiv(Cout, meta['BLOCK_N']),
        )
        conv1x1_batched_gemm_kernel[grid_batched](
            extracted, w, output,
            Cin, HW_pad, HW_out, Cout,
        )
        return output