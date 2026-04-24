import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: 1x1 conv2d followed by flatten (dims 2 onwards)
# in_0 = bias  [C_out]
# in_1 = weight [C_out, C_in, 1, 1]
# in_2 = input  [N, C_in, H, W]
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(conv2d, 2)
    return (tmp_3,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Fused Triton kernel:
#   output[n, co, l] = sum_ci( x[n, ci, l] * w[co, ci] ) + bias[co]
# where l = h*W + w is the flattened spatial index.
#
# Treats the whole batch×spatial as M = N*H*W rows, C_in as K cols,
# and C_out as N_out cols, then writes into the pre-flattened output shape.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 1024,'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N_out', 'K'],
)
@triton.jit
def fused_conv1x1_flatten_kernel(
    x_ptr,        # [N_batch, K, H, W]  NCHW layout
    w_ptr,        # [N_out, K]          (weight reshaped, 1x1 kernel)
    bias_ptr,     # [N_out]
    out_ptr,      # [N_batch, N_out, L] output, L = H*W
    M,            # N_batch * H * W
    N_out,        # C_out
    K,            # C_in
    HW,           # H * W
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ---- program ID decomposition ----------------------------------------
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_b = pid // num_pid_m   # batch index
    pid_m = pid % num_pid_m    # spatial-block index

    # ---- row/col offsets --------------------------------------------------
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_offs = tl.arange(0, BLOCK_N)                      # [BLOCK_N]

    m_mask = m_offs < M
    n_mask = n_offs < N_out

    # ---- base pointers for this batch element -----------------------------
    # x[n, ci, l] = x_ptr + n*K*HW + ci*HW + l
    x_batch_base = pid_b * K * HW

    # ---- accumulator -------------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---- main K loop -------------------------------------------------------
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k_start * BLOCK_K + tl.arange(0, BLOCK_K)   # [BLOCK_K]
        k_mask = k_offs < K

        # Load x tile: [BLOCK_M, BLOCK_K]
        # x[m, k] = x_ptr[x_batch_base + k*HW + m]
        x_ptrs = x_ptr + x_batch_base + k_offs[None, :] * HW + m_offs[:, None]
        x_vals = tl.load(x_ptrs,
                         mask=m_mask[:, None] & k_mask[None, :],
                         other=0.0)

        # Load w tile: [BLOCK_N, BLOCK_K]
        # w[n, k] = w_ptr + n*K + k
        w_ptrs = w_ptr + n_offs[:, None] * K + k_offs[None, :]
        w_vals = tl.load(w_ptrs,
                         mask=n_mask[:, None] & k_mask[None, :],
                         other=0.0)

        # acc += x_vals @ w_vals^T  -->  [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N]
        acc = tl.dot(x_vals, tl.trans(w_vals), acc)

    # ---- add bias ----------------------------------------------------------
    bias_vals = tl.load(bias_ptr + n_offs, mask=n_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # ---- store into output[N, N_out, L] ------------------------------------
    # out[n, co, l] = out_ptr + n*N_out*HW + co*HW + l
    out_batch_base = pid_b * N_out * HW
    out_ptrs = out_ptr + out_batch_base + n_offs[None, :] * HW + m_offs[:, None]
    tl.store(out_ptrs,
             acc.to(out_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Wrapper (must be @torch.fx.wrap so FX doesn't trace into it)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_conv1x1_flatten(bias, weight, x):
    """
    bias   : [C_out]
    weight : [C_out, C_in, 1, 1]
    x      : [N, C_in, H, W]
    returns: ([N, C_out, H*W],)
    """
    N_batch = x.shape[0]
    C_in    = x.shape[1]
    H       = x.shape[2]
    W       = x.shape[3]
    C_out   = weight.shape[0]
    HW      = H * W
    M       = N_batch * HW

    # Allocate output in flattened spatial layout [N, C_out, H*W]
    out = torch.empty((N_batch, C_out, HW), dtype=x.dtype, device=x.device)

    # weight[C_out, C_in, 1, 1] has strides [C_in, 1, 1, 1] — identical
    # memory layout to [C_out, C_in], so we pass it directly (no view needed).
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * N_batch,)

    fused_conv1x1_flatten_kernel[grid](
        x, weight, bias, out,
        M, C_out, C_in, HW,
    )

    return (out,)


def replacement_func():
    return fused_conv1x1_flatten