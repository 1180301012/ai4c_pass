import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: 1x1 conv2d  +  hardtanh(0,6)  +  element-wise multiply
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [C_out]
    in_1 : weight [C_out, C_in, 1, 1]
    in_2 : input  [N, C_in, H, W]
    in_3 : gate   [N, C_out, H, W]
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    tmp_4 = tmp_3 * conv2d
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Kernel 1 – matmul path: best for large N (N ≥ 8).
# Grid: (N, cdiv(HW, BLOCK_HW), cdiv(C_out, BLOCK_OC))
# Uses tl.dot / tensor-cores to maximise throughput.
# autotune key includes N so small-N and large-N get different tile choices.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # ---- num_stages=2: optimal for single K-iteration (C_in=24<BLOCK_K=32) ----
        triton.Config({'BLOCK_HW': 128, 'BLOCK_OC': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_OC': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_OC': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_OC': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        # ---- num_stages=4: more pipelining for float16 tensor-core path ----
        triton.Config({'BLOCK_HW': 128, 'BLOCK_OC': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_OC': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_OC': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_OC': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        # ---- Small tiles for N=1 (better GPU occupancy) ----
        triton.Config({'BLOCK_HW': 64,  'BLOCK_OC': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_HW': 64,  'BLOCK_OC': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_HW': 32,  'BLOCK_OC': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_HW': 32,  'BLOCK_OC': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['N', 'HW', 'C_out', 'C_in'],   # N included → small/large N tuned independently
)
@triton.jit
def _fused_conv1x1_relu6_mul_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr, out_ptr,
    N, HW, C_out, C_in,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_K:  tl.constexpr,
):
    """
    Fused 1×1 conv + hardtanh(0,6) on gate + element-wise multiply.
    Computes  out[oc, hw] = relu6(gate[oc, hw]) * (B @ A_T + bias)[oc, hw]
    where A_T[k,hw]=in_2[n,k,hw], B[oc,k]=weight[oc,k].

    ALL memory operations use a coalesced [BLOCK_OC, BLOCK_HW] layout
    (contiguous inner HW dimension, stride 1).  No register transposes.

    Layout (NCHW):
      A_T[k, hw] = A_base + k*HW + hw  → inner HW, stride 1 ✓
      B  [oc, k] = B_ptr  + oc*C_in+k  → inner K,  stride 1 ✓
      gate[oc,hw] = C_base + oc*HW + hw → inner HW, stride 1 ✓
      out [oc,hw] = out_base+oc*HW + hw → inner HW, stride 1 ✓
    """
    batch_n = tl.program_id(0)
    pid_hw  = tl.program_id(1)
    pid_oc  = tl.program_id(2)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k  = tl.arange(0, BLOCK_K)

    mask_hw = offs_hw < HW
    mask_oc = offs_oc < C_out

    A_base   = A_ptr   + batch_n * C_in  * HW
    C_base   = C_ptr   + batch_n * C_out * HW
    out_base = out_ptr + batch_n * C_out * HW

    # Load A_T as [BLOCK_K, BLOCK_HW]: inner hw stride 1 (coalesced) ✓
    a_t_ptrs = A_base + offs_k[:, None] * HW + offs_hw[None, :]
    # Load B as [BLOCK_OC, BLOCK_K]: inner k stride 1 (coalesced) ✓
    b_ptrs   = B_ptr  + offs_oc[:, None] * C_in + offs_k[None, :]

    # Accumulator in [BLOCK_OC, BLOCK_HW] layout (matches output layout)
    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for k_idx in range(0, tl.cdiv(C_in, BLOCK_K)):
        k_mask = (k_idx * BLOCK_K + offs_k) < C_in
        b   = tl.load(b_ptrs,   mask=mask_oc[:, None] & k_mask[None, :], other=0.0)
        a_t = tl.load(a_t_ptrs, mask=k_mask[:, None]  & mask_hw[None, :], other=0.0)
        # b:[BLOCK_OC, BLOCK_K] @ a_t:[BLOCK_K, BLOCK_HW] → [BLOCK_OC, BLOCK_HW]
        acc = tl.dot(b, a_t, acc)
        b_ptrs   += BLOCK_K        # next K chunk in B
        a_t_ptrs += BLOCK_K * HW   # next K chunk in A_T

    # ---- add bias: broadcast [BLOCK_OC] over BLOCK_HW ----
    bias_vals = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias_vals[:, None].to(tl.float32)

    # ---- load gate [BLOCK_OC, BLOCK_HW]: inner hw stride 1 ✓ ----
    c_ptrs  = C_base + offs_oc[:, None] * HW + offs_hw[None, :]
    gate    = tl.load(c_ptrs, mask=mask_oc[:, None] & mask_hw[None, :], other=0.0)
    gate_f32   = gate.to(tl.float32)
    gate_relu6 = tl.minimum(tl.maximum(gate_f32, 0.0), 6.0)

    # ---- fused multiply + store [BLOCK_OC, BLOCK_HW]: inner hw stride 1 ✓ ----
    result   = gate_relu6 * acc
    out_ptrs = out_base + offs_oc[:, None] * HW + offs_hw[None, :]
    tl.store(out_ptrs, result.to(gate.dtype),
             mask=mask_oc[:, None] & mask_hw[None, :])


# ---------------------------------------------------------------------------
# Kernel 2 – element-wise 1-D path: better for small N (N ≤ 8).
# Grid: (cdiv(TOTAL, BLOCK_S),)   where TOTAL = N*C_out*HW
# Each thread computes ONE output element via a scalar loop over C_in.
# More CUDA programs than the matmul path → better SM occupancy for N=1.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 128},  num_warps=4),
        triton.Config({'BLOCK_S': 256},  num_warps=4),
        triton.Config({'BLOCK_S': 512},  num_warps=8),
        triton.Config({'BLOCK_S': 1024}, num_warps=8),
    ],
    key=['TOTAL', 'C_in'],
)
@triton.jit
def _fused_conv1x1_relu6_mul_small_n_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr, out_ptr,
    TOTAL, C_in, C_out, HW,
    BLOCK_S: tl.constexpr,
):
    """
    Flat 1-D kernel.  Each lane handles one output element (n, oc, hw).
    The gate tensor (in_3) shares the same NCHW flat index as the output.
    A[n, ic, hw] = A_ptr + n*C_in*HW + ic*HW + hw  → contiguous in hw.
    B[oc, ic]    = B_ptr + oc*C_in   + ic           → contiguous in ic.
    """
    pid  = tl.program_id(0)
    offs = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = offs < TOTAL

    # Decode NCHW flat index → (n, oc, hw)
    n_idx  = offs // (C_out * HW)
    rem    = offs % (C_out * HW)
    oc_idx = rem  // HW
    hw_idx = rem  % HW

    # Pointers to the first C_in lane for this (n, hw) and (oc)
    A_base = A_ptr + n_idx * C_in * HW + hw_idx   # + ic*HW per channel
    B_base = B_ptr + oc_idx * C_in                 # + ic   per channel

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for ic in range(C_in):
        a_val = tl.load(A_base + ic * HW, mask=mask, other=0.0)
        b_val = tl.load(B_base + ic,      mask=mask, other=0.0)
        acc  += a_val.to(tl.float32) * b_val.to(tl.float32)

    bias_val = tl.load(bias_ptr + oc_idx, mask=mask, other=0.0)
    acc = acc + bias_val.to(tl.float32)

    # Gate is in_3, same NCHW layout as the output → flat index = offs
    gate      = tl.load(C_ptr + offs, mask=mask, other=0.0)
    gate_relu6 = tl.minimum(tl.maximum(gate.to(tl.float32), 0.0), 6.0)

    result = gate_relu6 * acc
    tl.store(out_ptr + offs, result.to(gate.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Weight cache: avoid repeated CPU→GPU transfers for constant model weights.
# Keyed by (CPU data_ptr, target device string, target dtype) so different
# dtype/device combinations are stored separately.
# ---------------------------------------------------------------------------
_WEIGHT_CACHE: dict = {}


def _get_weight(t: torch.Tensor, device, dtype) -> torch.Tensor:
    """Return t on *device* with *dtype*, using a cache to avoid re-transfers."""
    key = (t.data_ptr(), str(device), dtype)
    if key not in _WEIGHT_CACHE:
        _WEIGHT_CACHE[key] = t.to(device=device, dtype=dtype)
    return _WEIGHT_CACHE[key]


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def _fused_conv1x1_relu6_mul(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [C_out]              – may be on CPU
    in_1 : weight [C_out, C_in, 1, 1] – may be on CPU
    in_2 : input  [N, C_in, H, W]     – on CUDA
    in_3 : gate   [N, C_out, H, W]    – on CUDA
    """
    N, C_in, H, W = in_2.shape
    C_out = in_1.shape[0]
    HW    = H * W

    bias   = _get_weight(in_0, in_2.device, in_2.dtype)
    weight = _get_weight(in_1, in_2.device, in_2.dtype)

    out = torch.empty_like(in_3)

    if N <= 8:
        # Small-batch path: 1-D elementwise kernel creates more CUDA programs
        # for better SM occupancy on tiny workloads (N=1).
        TOTAL = N * C_out * HW
        grid_s = lambda meta: (triton.cdiv(TOTAL, meta['BLOCK_S']),)
        _fused_conv1x1_relu6_mul_small_n_kernel[grid_s](
            in_2, weight, bias, in_3, out,
            TOTAL, C_in, C_out, HW,
        )
    else:
        # Large-batch path: matmul (tl.dot / tensor-core) kernel.
        grid = lambda meta: (
            N,
            triton.cdiv(HW,    meta['BLOCK_HW']),
            triton.cdiv(C_out, meta['BLOCK_OC']),
        )
        _fused_conv1x1_relu6_mul_kernel[grid](
            in_2, weight, bias, in_3, out,
            N, HW, C_out, C_in,
        )

    return out


# ---------------------------------------------------------------------------
# Required interface functions
# ---------------------------------------------------------------------------

def replacement_func():
    return _fused_conv1x1_relu6_mul