import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: 1x1 conv followed by avg_pool2d(kernel=2, stride=2)
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    """
    Match: conv2d (1x1, no bias, stride=1, pad=0) -> avg_pool2d(2, 2, count_include_pad=True)
    NOTE: argument order must match model.py exactly (positional args).
    """
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.avg_pool2d(conv2d, 2, 2, 0, False, True, None)
    return tmp_2


def replacement_args(in_0, in_1):
    # in_0 = weight [C_out, C_in, 1, 1]
    # in_1 = input  [N, C_in, H, W]
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Two-kernel strategy  (pool input first, then GEMM on smaller tensor)
#
# Step 1: _avgpool2d_3d_kernel
#   [N, C_in, H, W] → pool_out [N, C_in, H/2, W/2]
#   • 3D grid: axis-0 = h_out (scalar, no division), axis-1 = NC blocks,
#              axis-2 = W blocks
#   • a00/a01 share a cache line; a10/a11 share a cache line
#
# Step 2: _conv1x1_3d_kernel
#   pool_out [N, C_in, H/2, W/2] × weight [C_out, C_in] → [N, C_out, H/2, W/2]
#   • 3D grid: axis-0 = batch (scalar, no batch crossings!),
#              axis-1 = h_out  (scalar, no division for spatial),
#              axis-2 = W_tiles × C_out_tiles (small product, minimal division)
#   • Each program stays within ONE batch × ONE h_out row → NO 100KB L2 jumps
#   • For fixed k, varying w: stride=1 (contiguous) → better cache efficiency
#   • fp16/bf16: native 16-bit tensor cores (125 TFLOPS on A30)
#   • fp32: TF32 tensor cores (10.3 TFLOPS)
# ---------------------------------------------------------------------------


# ---- Kernel 1: avg_pool2d via 3D grid (no integer division in hot path) ----

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_NC': 1,  'BLOCK_W': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_NC': 1,  'BLOCK_W': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_NC': 4,  'BLOCK_W': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_NC': 4,  'BLOCK_W': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_NC': 8,  'BLOCK_W': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_NC': 8,  'BLOCK_W': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_NC': 16, 'BLOCK_W': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_NC': 16, 'BLOCK_W': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_NC': 32, 'BLOCK_W': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_NC': 32, 'BLOCK_W': 32}, num_stages=4, num_warps=4),
    ],
    key=['NC', 'HW_out'],
)
@triton.jit
def _avgpool2d_3d_kernel(
    input_ptr, output_ptr,
    NC, H_out, W_out, W,
    HW_in,   # H * W
    HW_out,  # H_out * W_out
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_NC: tl.constexpr,
    BLOCK_W:  tl.constexpr,
):
    """
    3D grid: (H_out, NC_blocks, W_blocks).
    h_out from program_id(0) → NO division needed.
    """
    h_out    = tl.program_id(0)
    pid_nc   = tl.program_id(1)
    pid_w    = tl.program_id(2)

    nc_offs  = pid_nc * BLOCK_NC + tl.arange(0, BLOCK_NC)
    w_offs   = pid_w  * BLOCK_W  + tl.arange(0, BLOCK_W)

    nc_mask  = nc_offs < NC
    w_mask   = w_offs  < W_out
    mask     = nc_mask[:, None] & w_mask[None, :]

    h_in     = h_out * 2
    in_base  = nc_offs[:, None] * HW_in + h_in * W + w_offs[None, :] * 2

    a00 = tl.load(input_ptr + in_base,         mask=mask, other=0.0)
    a01 = tl.load(input_ptr + in_base + 1,     mask=mask, other=0.0)
    a10 = tl.load(input_ptr + in_base + W,     mask=mask, other=0.0)
    a11 = tl.load(input_ptr + in_base + W + 1, mask=mask, other=0.0)

    result = (a00.to(tl.float32) + a01.to(tl.float32)
              + a10.to(tl.float32) + a11.to(tl.float32)) * 0.25

    out_base = nc_offs[:, None] * HW_out + h_out * W_out + w_offs[None, :]

    if IS_FP16:
        tl.store(output_ptr + out_base, result.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(output_ptr + out_base, result.to(tl.bfloat16), mask=mask)
    else:
        tl.store(output_ptr + out_base, result, mask=mask)


# ---- Kernel 2: 1×1 conv on pooled input (3D GEMM, no batch crossings) ----
#
# Grid: (N_batch, H_out, ceil(W_out/BLOCK_W) × ceil(C_out/BLOCK_N))
#   batch = program_id(0)  → scalar, no division
#   h_out = program_id(1)  → scalar, no division
#   pid_wn= program_id(2)  → decompose into (pid_w, pid_n) with small division
#
# A-tile  [BLOCK_W, BLOCK_K]:
#   A[w, k] = pool_ptr + batch*s_n + k*HW_out + h_out*W_out + w
#   Fixed k, varying w: stride=1 (CONTIGUOUS) → perfect cache line use
#   Fixed w, varying k: stride=HW_out (same as before but no batch jumps)
#
# No tile ever crosses a batch boundary → eliminates 100KB L2 cache misses
# that hurt the flat-M approach for batch=32.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 16, 'BLOCK_N':  64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_W': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_W': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_W': 16, 'BLOCK_N':  64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_W': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_W': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_W': 32, 'BLOCK_N':  64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_W': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_W': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_W': 32, 'BLOCK_N':  64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_W': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_W': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_W': 64, 'BLOCK_N':  64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_W': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_W': 64, 'BLOCK_N':  64, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_W': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
    ],
    key=['HW_out', 'W_out', 'C_out', 'C_in'],
)
@triton.jit
def _conv1x1_3d_kernel(
    pool_ptr, weight_ptr, output_ptr,
    N_batch, C_out, C_in,
    H_out, W_out, HW_out,
    s_pool_n, s_pool_c, s_pool_h, s_pool_w,
    s_w_cout, s_w_cin,
    s_out_n, s_out_c, s_out_h, s_out_w,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ---- 3D program IDs: no batch crossings, no division for batch or h_out ----
    batch  = tl.program_id(0)   # which batch element  [0, N_batch)
    h_out  = tl.program_id(1)   # which output row     [0, H_out)
    pid_wn = tl.program_id(2)   # W-tile × N-tile combined index

    num_pid_w = tl.cdiv(W_out, BLOCK_W)
    pid_w = pid_wn % num_pid_w   # only small division (~1–3 range)
    pid_n = pid_wn // num_pid_w

    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)   # [BLOCK_W]
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    w_mask = w_offs < W_out
    n_mask = n_offs < C_out

    # Base address offset shared across K iterations (batch & h_out are scalar)
    base_addr = batch * s_pool_n + h_out * s_pool_h

    acc = tl.zeros((BLOCK_W, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, C_in, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)   # [BLOCK_K]

        # ---- Load weight tile B[BLOCK_K, BLOCK_N] ----
        w_ptrs = (weight_ptr
                  + n_offs[None, :] * s_w_cout
                  + k_offs[:, None] * s_w_cin)
        w_mask2 = (k_offs[:, None] < C_in) & n_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask2, other=0.0)   # [BLOCK_K, BLOCK_N]

        # ---- Load pool_out tile A[BLOCK_W, BLOCK_K] ----
        # A[w, k] = base_addr + k*s_pool_c + w*s_pool_w
        # For fixed k, w varies with stride s_pool_w=1 → CONTIGUOUS
        a_offs = (base_addr
                  + k_offs[None, :] * s_pool_c     # [1, BLOCK_K]
                  + w_offs[:, None] * s_pool_w)    # [BLOCK_W, 1]
        # → [BLOCK_W, BLOCK_K]
        inp_mask = w_mask[:, None] & (k_offs[None, :] < C_in)
        a = tl.load(pool_ptr + a_offs, mask=inp_mask, other=0.0)

        # ---- GEMM tile with native dtype tensor cores ----
        if IS_FP16:
            acc += tl.dot(a.to(tl.float16), w.to(tl.float16))
        elif IS_BF16:
            acc += tl.dot(a.to(tl.bfloat16), w.to(tl.bfloat16))
        else:
            acc += tl.dot(a.to(tl.float32), w.to(tl.float32), allow_tf32=True)

    # ---- Store output ----
    out_mask = w_mask[:, None] & n_mask[None, :]
    out_ptrs = (output_ptr
                + batch  * s_out_n
                + h_out  * s_out_h
                + w_offs[:, None] * s_out_w
                + n_offs[None, :] * s_out_c)

    if IS_FP16:
        tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)
    elif IS_BF16:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(out_ptrs, acc, mask=out_mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_conv1x1_avgpool(weight, x):
    """
    Two-kernel replacement for conv1x1 + avg_pool2d.

    weight : [C_out, C_in, 1, 1]
    x      : [N, C_in, H, W]
    returns: [N, C_out, H//2, W//2]
    """
    N, C_in, H, W = x.shape
    C_out  = weight.shape[0]
    H_out  = H // 2
    W_out  = W // 2
    HW_in  = H * W
    HW_out = H_out * W_out
    NC     = N * C_in

    IS_FP16 = (x.dtype == torch.float16)
    IS_BF16 = (x.dtype == torch.bfloat16)

    # ---- Step 1: pool the input ----
    pool_out = torch.empty((N, C_in, H_out, W_out), device=x.device, dtype=x.dtype)

    def pool_grid(META):
        return (H_out,
                triton.cdiv(NC,   META['BLOCK_NC']),
                triton.cdiv(W_out, META['BLOCK_W']))

    _avgpool2d_3d_kernel[pool_grid](
        x, pool_out,
        NC, H_out, W_out, W,
        HW_in, HW_out,
        IS_FP16=IS_FP16, IS_BF16=IS_BF16,
    )

    # ---- Step 2: 1×1 conv (3D GEMM) on pooled input ----
    output = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    s_pool_n = pool_out.stride(0)   # C_in * H_out * W_out
    s_pool_c = pool_out.stride(1)   # H_out * W_out
    s_pool_h = pool_out.stride(2)   # W_out
    s_pool_w = pool_out.stride(3)   # 1

    s_w_cout = weight.stride(0)     # C_in
    s_w_cin  = weight.stride(1)     # 1

    s_out_n  = output.stride(0)
    s_out_c  = output.stride(1)
    s_out_h  = output.stride(2)
    s_out_w  = output.stride(3)

    def gemm_grid(META):
        num_pid_w = triton.cdiv(W_out, META['BLOCK_W'])
        num_pid_n = triton.cdiv(C_out, META['BLOCK_N'])
        return (N, H_out, num_pid_w * num_pid_n)

    _conv1x1_3d_kernel[gemm_grid](
        pool_out, weight, output,
        N, C_out, C_in, H_out, W_out, HW_out,
        s_pool_n, s_pool_c, s_pool_h, s_pool_w,
        s_w_cout, s_w_cin,
        s_out_n, s_out_c, s_out_h, s_out_w,
        IS_FP16=IS_FP16, IS_BF16=IS_BF16,
    )

    return output


# ---------------------------------------------------------------------------
# replacement_func must return a callable (not call it)
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_conv1x1_avgpool
#       Pool the input [N, C_in, H, W] → pool_out [N, C_in, H/2, W/2]
#       • 3D grid: (H_out, ceil(N*C_in/BLOCK_NC), ceil(W_out/BLOCK_W))
#         - h_out from program_id(0)  → NO integer division inside kernel
#         - nc_offs from program_id(1)
#         - w_offs from program_id(2)
#       • Loads: stride-2 pairs (a00/a01 share cache line; a10/a11 share cache line)
#
#   Step 2 (_conv1x1_nchw_kernel):
#       1×1 conv on pooled input  (GEMM)
#       pool_out [N, C_in, H/2, W/2] × weight [C_out, C_in] → output [N, C_out, H/2, W/2]
#       • K-stride = H_out*W_out  (4× smaller than raw input K-stride H*W)
#       • fp16/bf16: use native tensor cores (125 TFLOPS on A30)
#       • fp32: TF32 tensor cores (10.3 TFLOPS)
#       • Small BLOCK_M configs (16/32) for better SM utilization at small M
# ---------------------------------------------------------------------------


# ---- Kernel 1: avg_pool2d via 3D grid (no integer division in hot path) ----

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_NC': 1,  'BLOCK_W': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_NC': 1,  'BLOCK_W': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_NC': 4,  'BLOCK_W': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_NC': 4,  'BLOCK_W': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_NC': 8,  'BLOCK_W': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_NC': 8,  'BLOCK_W': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_NC': 16, 'BLOCK_W': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_NC': 16, 'BLOCK_W': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_NC': 32, 'BLOCK_W': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_NC': 32, 'BLOCK_W': 32}, num_stages=4, num_warps=4),
    ],
    key=['NC', 'HW_out'],
)
@triton.jit
def _avgpool2d_3d_kernel(
    input_ptr, output_ptr,
    NC, H_out, W_out, W,
    HW_in,   # H * W
    HW_out,  # H_out * W_out
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_NC: tl.constexpr,
    BLOCK_W:  tl.constexpr,   # power-of-2, >= W_out handled by mask
):
    """
    3D grid: axis-0 = h_out (no division needed!), axis-1 = nc blocks, axis-2 = w blocks.

    For each output position (nc, h_out, w_out):
      output[nc, h_out, w_out] = avg of 4 input pixels at
        (nc, 2*h_out, 2*w_out), (+1 in W), (+1 in H), (+1 in both)
    """
    h_out    = tl.program_id(0)           # exact h_out — no division needed!
    pid_nc   = tl.program_id(1)
    pid_w    = tl.program_id(2)

    nc_offs  = pid_nc * BLOCK_NC + tl.arange(0, BLOCK_NC)   # [BLOCK_NC]
    w_offs   = pid_w  * BLOCK_W  + tl.arange(0, BLOCK_W)    # [BLOCK_W]

    nc_mask  = nc_offs < NC
    w_mask   = w_offs  < W_out
    mask     = nc_mask[:, None] & w_mask[None, :]            # [BLOCK_NC, BLOCK_W]

    h_in     = h_out * 2

    # Input base: nc * HW_in + h_in * W + w_out * 2   (no modulo/division!)
    in_base  = (nc_offs[:, None] * HW_in
                + h_in * W
                + w_offs[None, :] * 2)                       # [BLOCK_NC, BLOCK_W]

    # 4 reads: +0/+1 share a cache line; +W/+W+1 share a cache line
    a00 = tl.load(input_ptr + in_base,         mask=mask, other=0.0)
    a01 = tl.load(input_ptr + in_base + 1,     mask=mask, other=0.0)
    a10 = tl.load(input_ptr + in_base + W,     mask=mask, other=0.0)
    a11 = tl.load(input_ptr + in_base + W + 1, mask=mask, other=0.0)

    result = (a00.to(tl.float32) + a01.to(tl.float32)
              + a10.to(tl.float32) + a11.to(tl.float32)) * 0.25

    # Output base: nc * HW_out + h_out * W_out + w_out  (stride-1 in W → coalesced)
    out_base = (nc_offs[:, None] * HW_out
                + h_out * W_out
                + w_offs[None, :])                           # [BLOCK_NC, BLOCK_W]

    if IS_FP16:
        tl.store(output_ptr + out_base, result.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(output_ptr + out_base, result.to(tl.bfloat16), mask=mask)
    else:
        tl.store(output_ptr + out_base, result, mask=mask)


# ---- Kernel 2: 1×1 conv on pooled NCHW input (GEMM) ----
#
# GEMM view:
#   A[m, k] = pool_out[batch, k, h_out, w_out]  (K-stride = HW_out, 4× smaller than H*W)
#   B[k, n] = weight[n, k]                       (K contiguous, stride=1)
#   output[m, n] → [batch, n, h_out, w_out]
#
# fp16/bf16 → native tensor cores; fp32 → TF32 tensor cores
# Small BLOCK_M (16/32) configs added for better SM utilization at small M

@triton.autotune(
    configs=[
        # ---- Small M configs (batch=1 or small batch) ----
        triton.Config({'BLOCK_M':  16, 'BLOCK_N':  64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  32, 'BLOCK_N':  64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  32, 'BLOCK_N':  64, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        # ---- Standard M configs ----
        triton.Config({'BLOCK_M':  64, 'BLOCK_N':  64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N':  64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        # ---- Large M configs ----
        triton.Config({'BLOCK_M': 256, 'BLOCK_N':  64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
    ],
    key=['M', 'C_out', 'C_in'],
)
@triton.jit
def _conv1x1_nchw_kernel(
    pool_ptr, weight_ptr, output_ptr,
    M, C_out, C_in,
    H_out, W_out,
    s_pool_n, s_pool_c, s_pool_h, s_pool_w,
    s_w_cout, s_w_cin,
    s_out_n, s_out_c, s_out_h, s_out_w,
    HW_out,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # ---- L2-cache-friendly swizzled program ordering ----
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(C_out, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id      = pid // num_pid_in_group
    first_pid_m   = group_id * GROUP_M
    group_size_m  = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    # Decode m → (batch, h_out, w_out)
    batch_idx  = m_offs // HW_out
    spatial    = m_offs % HW_out
    h_out_idx  = spatial // W_out
    w_out_idx  = spatial % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, C_in, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)   # [BLOCK_K]

        # ---- Load weight tile B[BLOCK_K, BLOCK_N]: B[k,n] = weight[n,k] ----
        w_ptrs = (weight_ptr
                  + n_offs[None, :] * s_w_cout
                  + k_offs[:, None] * s_w_cin)
        w_mask = (k_offs[:, None] < C_in) & (n_offs[None, :] < C_out)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)    # [BLOCK_K, BLOCK_N]

        # ---- Load pool_out tile A[BLOCK_M, BLOCK_K] (single load) ----
        inp_mask = (m_offs[:, None] < M) & (k_offs[None, :] < C_in)
        a_offs = (batch_idx[:, None] * s_pool_n
                  + k_offs[None,  :] * s_pool_c
                  + h_out_idx[:,  None] * s_pool_h
                  + w_out_idx[:,  None] * s_pool_w)    # [BLOCK_M, BLOCK_K]
        a = tl.load(pool_ptr + a_offs, mask=inp_mask, other=0.0)

        # ---- GEMM tile with native dtype tensor cores ----
        if IS_FP16:
            acc += tl.dot(a.to(tl.float16), w.to(tl.float16))
        elif IS_BF16:
            acc += tl.dot(a.to(tl.bfloat16), w.to(tl.bfloat16))
        else:
            acc += tl.dot(a.to(tl.float32), w.to(tl.float32), allow_tf32=True)

    # ---- Store output ----
    m_mask   = m_offs < M
    n_mask   = n_offs < C_out
    out_mask = m_mask[:, None] & n_mask[None, :]

    out_ptrs = (output_ptr
                + batch_idx[:, None] * s_out_n
                + n_offs[None, :]    * s_out_c
                + h_out_idx[:, None] * s_out_h
                + w_out_idx[:, None] * s_out_w)

    if IS_FP16:
        tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)
    elif IS_BF16:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(out_ptrs, acc, mask=out_mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_conv1x1_avgpool(weight, x):
    """
    Two-kernel replacement for conv1x1 + avg_pool2d.

    weight : [C_out, C_in, 1, 1]
    x      : [N, C_in, H, W]
    returns: [N, C_out, H//2, W//2]

    Step 1: pool x → pool_out [N, C_in, H//2, W//2]
            3D grid avoids integer division; stride-2 pairs share cache lines.
    Step 2: GEMM on pool_out with K-stride = H_out*W_out (4× smaller than H*W)
            + native fp16/bf16 tensor cores for lower-precision dtypes.
    """
    N, C_in, H, W = x.shape
    C_out  = weight.shape[0]
    H_out  = H // 2
    W_out  = W // 2
    M      = N * H_out * W_out
    HW_in  = H * W
    HW_out = H_out * W_out
    NC     = N * C_in

    IS_FP16 = (x.dtype == torch.float16)
    IS_BF16 = (x.dtype == torch.bfloat16)

    # ---- Step 1: pool the input (3D grid) ----
    pool_out = torch.empty((N, C_in, H_out, W_out), device=x.device, dtype=x.dtype)

    def pool_grid(META):
        return (H_out,
                triton.cdiv(NC, META['BLOCK_NC']),
                triton.cdiv(W_out, META['BLOCK_W']))

    _avgpool2d_3d_kernel[pool_grid](
        x, pool_out,
        NC, H_out, W_out, W,
        HW_in, HW_out,
        IS_FP16=IS_FP16, IS_BF16=IS_BF16,
    )

    # ---- Step 2: 1×1 conv (GEMM) on pooled input ----
    output = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    s_pool_n = pool_out.stride(0)   # C_in * H_out * W_out
    s_pool_c = pool_out.stride(1)   # H_out * W_out
    s_pool_h = pool_out.stride(2)   # W_out
    s_pool_w = pool_out.stride(3)   # 1

    s_w_cout = weight.stride(0)     # C_in
    s_w_cin  = weight.stride(1)     # 1

    s_out_n  = output.stride(0)
    s_out_c  = output.stride(1)
    s_out_h  = output.stride(2)
    s_out_w  = output.stride(3)

    def gemm_grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(C_out, META['BLOCK_N']),)

    _conv1x1_nchw_kernel[gemm_grid](
        pool_out, weight, output,
        M, C_out, C_in, H_out, W_out,
        s_pool_n, s_pool_c, s_pool_h, s_pool_w,
        s_w_cout, s_w_cin,
        s_out_n, s_out_c, s_out_h, s_out_w,
        HW_out,
        IS_FP16=IS_FP16, IS_BF16=IS_BF16,
    )

    return output


# ---------------------------------------------------------------------------
# replacement_func must return a callable (not call it)
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_conv1x1_avgpool


# ---------------------------------------------------------------------------
# Two-kernel strategy:
#
#   Step 1 (_avgpool2d_kernel):
#       Pool the input [N, C_in, H, W] → pool_out [N, C_in, H/2, W/2]
#       • Stride-1 sequential reads (each input row read once, not 4×)
#       • Reduces K-stride from H*W → H_out*W_out (4× smaller) for Step 2
#
#   Step 2 (_conv1x1_nchw_kernel):
#       1×1 conv on pooled input  (standard GEMM)
#       pool_out [N, C_in, H/2, W/2]  ×  weight [C_out, C_in]
#       → output [N, C_out, H/2, W/2]
#       • fp16/bf16 inputs use native tensor cores (125 TFLOPS on A30)
#       • fp32 inputs use TF32 tensor cores (10.3 TFLOPS)
#
# Total memory:
#   input (1×) + pool_out (1×) + weight + output
#   vs cuDNN: input (1×) + weight + conv_intermediate (1× write + 1× read) + output
#   We save the conv_intermediate write/read (≈ N*C_out*H*W bytes).
# ---------------------------------------------------------------------------


# ---- Kernel 1: avg_pool2d(kernel=2, stride=2) on NCHW input ----

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=8),
    ],
    key=['N_total'],
)
@triton.jit
def _avgpool2d_kernel(
    input_ptr, output_ptr,
    N_total,          # N * C_in * H_out * W_out  (total output elements)
    H_out, W_out, W,  # spatial dims
    s_nc,             # input stride per (n,c) slice = H * W
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Flat kernel: each program handles BLOCK_SIZE output elements.
    Decodes flat idx → (nc, h_out, w_out) and loads 4 input pixels.
    The +0/+1 pair and +W/+W+1 pair within each row are adjacent in
    memory → near-sequential cache lines despite stride-2 access.
    """
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_total

    HW_out   = H_out * W_out
    nc       = offs // HW_out
    hw_rem   = offs % HW_out
    h_out    = hw_rem // W_out
    w_out    = hw_rem % W_out

    base = nc * s_nc + h_out * 2 * W + w_out * 2

    a00 = tl.load(input_ptr + base,         mask=mask, other=0.0)
    a01 = tl.load(input_ptr + base + 1,     mask=mask, other=0.0)
    a10 = tl.load(input_ptr + base + W,     mask=mask, other=0.0)
    a11 = tl.load(input_ptr + base + W + 1, mask=mask, other=0.0)

    result = (a00.to(tl.float32) + a01.to(tl.float32)
              + a10.to(tl.float32) + a11.to(tl.float32)) * 0.25

    if IS_FP16:
        tl.store(output_ptr + offs, result.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(output_ptr + offs, result.to(tl.bfloat16), mask=mask)
    else:
        tl.store(output_ptr + offs, result, mask=mask)


# ---- Kernel 2: 1×1 conv on pooled NCHW input (GEMM) ----
#
# GEMM view  (M = N*H_out*W_out, K = C_in, N_out = C_out):
#   A[m, k] = pool_out[batch, k, h_out, w_out]
#             stride in K = s_pool_c = H_out*W_out  (4× smaller than raw input)
#   B[k, n] = weight[n, k]          (contiguous in K since weight.stride(1)=1)
#   C[m, n] → output[batch, n, h_out, w_out]
#
# Key improvement over the original fused kernel:
#   • 1 load per (m,k) instead of 4
#   • K-stride reduced from H*W to H_out*W_out (better L1/L2 hit rate)
#   • fp16/bf16 inputs use 16-bit tensor cores (125 TFLOPS vs 10 TFLOPS TF32)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M':  64, 'BLOCK_N':  64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N':  64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N':  64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
    ],
    key=['M', 'C_out', 'C_in'],
)
@triton.jit
def _conv1x1_nchw_kernel(
    pool_ptr, weight_ptr, output_ptr,
    M, C_out, C_in,
    H_out, W_out,
    s_pool_n, s_pool_c, s_pool_h, s_pool_w,
    s_w_cout, s_w_cin,
    s_out_n, s_out_c, s_out_h, s_out_w,
    HW_out,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # ---- L2-cache-friendly swizzled program ordering ----
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(C_out, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id      = pid // num_pid_in_group
    first_pid_m   = group_id * GROUP_M
    group_size_m  = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    # Decode m → (batch, h_out, w_out)
    batch_idx  = m_offs // HW_out
    spatial    = m_offs % HW_out
    h_out_idx  = spatial // W_out
    w_out_idx  = spatial % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, C_in, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)   # [BLOCK_K]

        # ---- Load weight tile  B[BLOCK_K, BLOCK_N]: B[k,n] = weight[n,k] ----
        w_ptrs = (weight_ptr
                  + n_offs[None, :] * s_w_cout
                  + k_offs[:, None] * s_w_cin)
        w_mask = (k_offs[:, None] < C_in) & (n_offs[None, :] < C_out)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)    # [BLOCK_K, BLOCK_N]

        # ---- Load pool_out tile  A[BLOCK_M, BLOCK_K]  (single load per elem) ----
        inp_mask = (m_offs[:, None] < M) & (k_offs[None, :] < C_in)
        a_offs = (batch_idx[:, None] * s_pool_n
                  + k_offs[None,  :] * s_pool_c
                  + h_out_idx[:,  None] * s_pool_h
                  + w_out_idx[:,  None] * s_pool_w)    # [BLOCK_M, BLOCK_K]
        a = tl.load(pool_ptr + a_offs, mask=inp_mask, other=0.0)

        # ---- GEMM tile with native dtype tensor cores ----
        if IS_FP16:
            # fp16 tensor cores: 125 TFLOPS on A30, fp32 accumulation
            acc += tl.dot(a.to(tl.float16), w.to(tl.float16))
        elif IS_BF16:
            # bf16 tensor cores: 125 TFLOPS on A30, fp32 accumulation
            acc += tl.dot(a.to(tl.bfloat16), w.to(tl.bfloat16))
        else:
            # TF32 tensor cores for fp32: 10.3 TFLOPS on A30
            acc += tl.dot(a.to(tl.float32), w.to(tl.float32), allow_tf32=True)

    # ---- Store output ----
    m_mask   = m_offs < M
    n_mask   = n_offs < C_out
    out_mask = m_mask[:, None] & n_mask[None, :]

    out_ptrs = (output_ptr
                + batch_idx[:, None] * s_out_n
                + n_offs[None, :]    * s_out_c
                + h_out_idx[:, None] * s_out_h
                + w_out_idx[:, None] * s_out_w)

    if IS_FP16:
        tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)
    elif IS_BF16:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(out_ptrs, acc, mask=out_mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_conv1x1_avgpool(weight, x):
    """
    Two-kernel replacement for conv1x1 + avg_pool2d.

    weight : [C_out, C_in, 1, 1]
    x      : [N, C_in, H, W]
    returns: [N, C_out, H//2, W//2]

    Step 1: pool x → pool_out  [N, C_in, H//2, W//2]
            (reads input once; K-stride 4× smaller for Step 2)
    Step 2: 1×1-conv on pool_out using native TC dtype GEMM
            (avoids 4× input reads; uses fp16/bf16 tensor cores)
    """
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    H_out = H // 2
    W_out = W // 2
    M        = N * H_out * W_out
    HW_out   = H_out * W_out
    N_total  = N * C_in * H_out * W_out   # pool output elements

    IS_FP16 = (x.dtype == torch.float16)
    IS_BF16 = (x.dtype == torch.bfloat16)

    # ---- Step 1: pool the input ----
    pool_out = torch.empty((N, C_in, H_out, W_out), device=x.device, dtype=x.dtype)

    def pool_grid(META):
        return (triton.cdiv(N_total, META['BLOCK_SIZE']),)

    _avgpool2d_kernel[pool_grid](
        x, pool_out,
        N_total, H_out, W_out, W,
        H * W,                 # s_nc = H*W  (stride per (n,c) slice in [N*C_in,H,W])
        IS_FP16=IS_FP16, IS_BF16=IS_BF16,
    )

    # ---- Step 2: 1×1 conv (GEMM) on pooled input ----
    output = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    s_pool_n = pool_out.stride(0)   # C_in * H_out * W_out
    s_pool_c = pool_out.stride(1)   # H_out * W_out
    s_pool_h = pool_out.stride(2)   # W_out
    s_pool_w = pool_out.stride(3)   # 1

    s_w_cout = weight.stride(0)     # C_in
    s_w_cin  = weight.stride(1)     # 1

    s_out_n  = output.stride(0)
    s_out_c  = output.stride(1)
    s_out_h  = output.stride(2)
    s_out_w  = output.stride(3)

    def gemm_grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(C_out, META['BLOCK_N']),)

    _conv1x1_nchw_kernel[gemm_grid](
        pool_out, weight, output,
        M, C_out, C_in, H_out, W_out,
        s_pool_n, s_pool_c, s_pool_h, s_pool_w,
        s_w_cout, s_w_cin,
        s_out_n, s_out_c, s_out_h, s_out_w,
        HW_out,
        IS_FP16=IS_FP16, IS_BF16=IS_BF16,
    )

    return output


# ---------------------------------------------------------------------------
# replacement_func must return a callable (not call it)
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_conv1x1_avgpool