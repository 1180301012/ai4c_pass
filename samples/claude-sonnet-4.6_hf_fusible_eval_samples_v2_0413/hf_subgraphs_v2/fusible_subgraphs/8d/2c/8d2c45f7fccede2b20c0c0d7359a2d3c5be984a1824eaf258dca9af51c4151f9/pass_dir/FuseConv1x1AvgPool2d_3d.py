import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: 1x1 conv followed by avg_pool2d(kernel=2, stride=2)
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.avg_pool2d(conv2d, 2, 2, 0, False, True, None)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Step 1: avg_pool2d via 3D grid  [N,C_in,H,W] → [N,C_in,H/2,W/2]
#   axis-0 = h_out  (scalar, NO division needed)
#   axis-1 = NC blocks
#   axis-2 = W blocks
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_NC': 2,  'BLOCK_W': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_NC': 2,  'BLOCK_W': 32}, num_stages=4, num_warps=4),
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
def _pool2d_3d(
    inp_ptr, out_ptr,
    NC, H_out, W_out, W,
    HW_in, HW_out,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_NC: tl.constexpr,
    BLOCK_W:  tl.constexpr,
):
    h_out  = tl.program_id(0)          # exact h_out value — no division!
    pid_nc = tl.program_id(1)
    pid_w  = tl.program_id(2)

    nc_offs = pid_nc * BLOCK_NC + tl.arange(0, BLOCK_NC)
    w_offs  = pid_w  * BLOCK_W  + tl.arange(0, BLOCK_W)

    mask = (nc_offs[:, None] < NC) & (w_offs[None, :] < W_out)

    h_in    = h_out * 2
    base    = nc_offs[:, None] * HW_in + h_in * W + w_offs[None, :] * 2

    a00 = tl.load(inp_ptr + base,         mask=mask, other=0.0)
    a01 = tl.load(inp_ptr + base + 1,     mask=mask, other=0.0)
    a10 = tl.load(inp_ptr + base + W,     mask=mask, other=0.0)
    a11 = tl.load(inp_ptr + base + W + 1, mask=mask, other=0.0)

    avg = (a00.to(tl.float32) + a01.to(tl.float32)
           + a10.to(tl.float32) + a11.to(tl.float32)) * 0.25

    out_base = nc_offs[:, None] * HW_out + h_out * W_out + w_offs[None, :]
    if IS_FP16:
        tl.store(out_ptr + out_base, avg.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(out_ptr + out_base, avg.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + out_base, avg, mask=mask)


# ---------------------------------------------------------------------------
# Step 2: 1x1 conv (1D flat GEMM) on pooled input
#   M = N * H_out * W_out  (flat spatial dimension)
#   A[m, k] = pool_out[batch, k, h_out, w_out]  (K-stride = HW_out)
#   B[k, n] = weight[n, k]                       (K contiguous)
#   • Small BLOCK_M (16/32): more tiles for better SM utilization at small M
#   • Large BLOCK_M (128-256): high arithmetic intensity for large M
#   • fp16/bf16: native tensor cores; fp32: TF32
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # ---- Small M (batch=1 or batch=32) ----
        triton.Config({'BLOCK_M':  16, 'BLOCK_N':  64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  32, 'BLOCK_N':  64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  32, 'BLOCK_N':  64, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        # ---- Medium/large M ----
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
def _gemm_1d(
    pool_ptr, wt_ptr, out_ptr,
    M, C_out, C_in,
    H_out, W_out, HW_out,
    sp_n, sp_c, sp_h, sp_w,
    sw_n, sw_k,
    so_n, so_c, so_h, so_w,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(C_out, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id      = pid // num_pid_in_group
    first_pid_m   = group_id * GROUP_M
    group_size_m  = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    batch_idx = m_offs // HW_out
    spatial   = m_offs % HW_out
    h_out_idx = spatial // W_out
    w_out_idx = spatial % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, C_in, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        b_ptrs = wt_ptr + n_offs[None, :] * sw_n + k_offs[:, None] * sw_k
        b_mask = (k_offs[:, None] < C_in) & (n_offs[None, :] < C_out)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        a_offs = (batch_idx[:, None] * sp_n
                  + k_offs[None,  :] * sp_c
                  + h_out_idx[:, None] * sp_h
                  + w_out_idx[:, None] * sp_w)
        a_mask = (m_offs[:, None] < M) & (k_offs[None, :] < C_in)
        a = tl.load(pool_ptr + a_offs, mask=a_mask, other=0.0)

        if IS_FP16:
            acc += tl.dot(a.to(tl.float16), b.to(tl.float16))
        elif IS_BF16:
            acc += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
        else:
            acc += tl.dot(a.to(tl.float32), b.to(tl.float32), allow_tf32=True)

    m_mask = m_offs < M
    n_mask = n_offs < C_out
    out_mask = m_mask[:, None] & n_mask[None, :]
    o_ptrs = (out_ptr
              + batch_idx[:, None] * so_n
              + n_offs[None, :]    * so_c
              + h_out_idx[:, None] * so_h
              + w_out_idx[:, None] * so_w)
    if IS_FP16:
        tl.store(o_ptrs, acc.to(tl.float16), mask=out_mask)
    elif IS_BF16:
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(o_ptrs, acc, mask=out_mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_conv1x1_avgpool_3d(weight, x):
    N, C_in, H, W = x.shape
    C_out  = weight.shape[0]
    H_out  = H // 2
    W_out  = W // 2
    HW_in  = H * W
    HW_out = H_out * W_out
    NC     = N * C_in

    IS_FP16 = (x.dtype == torch.float16)
    IS_BF16 = (x.dtype == torch.bfloat16)

    # ---- Pool input ----
    pool = torch.empty((N, C_in, H_out, W_out), device=x.device, dtype=x.dtype)

    def pool_grid(META):
        return (H_out,
                triton.cdiv(NC, META['BLOCK_NC']),
                triton.cdiv(W_out, META['BLOCK_W']))

    _pool2d_3d[pool_grid](
        x, pool,
        NC, H_out, W_out, W,
        HW_in, HW_out,
        IS_FP16=IS_FP16, IS_BF16=IS_BF16,
    )

    # ---- 1D flat GEMM on pooled input ----
    M = N * H_out * W_out
    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    def gemm_grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(C_out, META['BLOCK_N']),)

    _gemm_1d[gemm_grid](
        pool, weight, out,
        M, C_out, C_in, H_out, W_out, HW_out,
        pool.stride(0), pool.stride(1), pool.stride(2), pool.stride(3),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        IS_FP16=IS_FP16, IS_BF16=IS_BF16,
    )

    return out


def replacement_func():
    return fused_conv1x1_avgpool_3d