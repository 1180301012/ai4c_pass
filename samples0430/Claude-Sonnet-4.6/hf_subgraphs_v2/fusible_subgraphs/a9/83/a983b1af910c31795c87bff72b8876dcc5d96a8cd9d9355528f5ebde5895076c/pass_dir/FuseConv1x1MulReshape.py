import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: 1x1 conv2d  +  (* 1.0 no-op)  +  reshape(-1, 17, 4096)
# ──────────────────────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d * 1.0
    tmp_4 = tmp_3.reshape(-1, 17, 4096)
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2):
    # in_0 = bias  [17]
    # in_1 = weight [17, 256, 1, 1]
    # in_2 = input  [N, 256, 64, 64]
    return (in_0, in_1, in_2)


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused 1x1-conv (GEMM) + bias + reshape
#
#   Treats the computation as:
#     A  = input  viewed as [M, IC]  where M = N * HW,  HW = H*W
#     B  = weight viewed as [OC, IC]
#     out = A @ B^T + bias          → [M, OC]
#   then stores in [N, OC, HW] memory order (= reshape(-1, OC, HW))
#
#   Memory layout of input (NCHW):
#     input[n, ic, hw] = input_ptr + n*IC*HW + ic*HW + hw
#   For m = n*HW + hw:
#     input[m, ic] = input_ptr + (m//HW)*IC*HW + ic*HW + (m%HW)
#   Row stride (varying m, fixed ic) = 1  →  coalesced ✓
#   Col stride (varying ic, fixed m) = HW →  handled by L1 / prefetch
#
#   Output (NCHW-reshaped → [N, OC, HW]):
#     out[n, oc, hw] = out_ptr + n*OC*HW + oc*HW + hw
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        # Large batch configs
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 64,  'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 128, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 128, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64,  'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32,  'num_stages': 4, 'num_warps': 4}),
        # Smaller batch / low-latency configs
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 32,  'num_stages': 5, 'num_warps': 2}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 128, 'num_stages': 3, 'num_warps': 4}),
    ],
    key=['M_total', 'IC', 'OC'],
)
@triton.jit
def _conv1x1_fused_kernel(
    x_ptr,     # input  [N, IC, HW] (NCHW flattened spatial)
    w_ptr,     # weight [OC, IC]
    bias_ptr,  # bias   [OC]
    out_ptr,   # output [N, OC, HW]
    M_total,   # N * HW
    IC,        # input channels  (256)
    OC,        # output channels (17)
    HW,        # H * W           (4096)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    offs_m = m0 + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_n = n0 + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    # Decode flat m-index into batch and spatial (hw) positions
    batch = offs_m // HW                   # [BLOCK_M]
    hw    = offs_m %  HW                   # [BLOCK_M]

    # Accumulate in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, IC, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)   # [BLOCK_K]

        # ── Load input tile [BLOCK_M, BLOCK_K] ──────────────────────────────
        # x[n, ic, hw] = x_ptr + n*IC*HW + ic*HW + hw
        x_ptrs = (x_ptr
                  + batch[:, None] * (IC * HW)
                  + offs_k[None, :] * HW
                  + hw[:, None])
        x_mask = (offs_m[:, None] < M_total) & (offs_k[None, :] < IC)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

        # ── Load weight tile [BLOCK_N, BLOCK_K] ─────────────────────────────
        # w[oc, ic] = w_ptr + oc*IC + ic
        w_ptrs = w_ptr + offs_n[:, None] * IC + offs_k[None, :]
        w_mask = (offs_n[:, None] < OC) & (offs_k[None, :] < IC)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # ── Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] ────────────
        acc = tl.dot(x_tile, tl.trans(w_tile), acc)

    # Add bias [OC]
    bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < OC, other=0.0).to(tl.float32)
    acc += bias_vals[None, :]

    # ── Store output [N, OC, HW]: out[n, oc, hw] = out_ptr + n*OC*HW + oc*HW + hw
    out_ptrs = (out_ptr
                + batch[:, None] * (OC * HW)
                + offs_n[None, :] * HW
                + hw[:, None])
    out_mask = (offs_m[:, None] < M_total) & (offs_n[None, :] < OC)
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper (must be @torch.fx.wrap)
# ──────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def conv1x1_fused(bias, weight, x):
    """
    Replaces:  conv2d(x, weight, bias, stride=1, pad=0, dil=1, groups=1)
                   * 1.0
                   .reshape(-1, OC, HW)
    weight shape: [OC, IC, 1, 1] — same memory layout as [OC, IC], last two dims = 1
    """
    N, IC, H, W = x.shape
    OC   = weight.shape[0]    # 17
    HW   = H * W              # 4096
    M_total = N * HW

    # Allocate output in the final [N, OC, HW] layout
    # This is identical to reshape(-1, OC, HW) since the batch dim is preserved
    out = torch.empty((N, OC, HW), dtype=x.dtype, device=x.device)

    # NOTE: weight is [OC, IC, 1, 1]; memory layout identical to [OC, IC] since kH=kW=1.
    #       We pass weight directly and index as w_ptr + oc*IC + ic in the kernel.

    grid = lambda meta: (
        triton.cdiv(M_total, meta['BLOCK_M']),
        triton.cdiv(OC,      meta['BLOCK_N']),
    )

    _conv1x1_fused_kernel[grid](
        x, weight, bias, out,
        M_total, IC, OC, HW,
    )

    # out is already [N, OC, HW] == [-1, OC, HW]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# replacement_func: zero-arg, returns callable
# ──────────────────────────────────────────────────────────────────────────────

def replacement_func():
    return conv1x1_fused