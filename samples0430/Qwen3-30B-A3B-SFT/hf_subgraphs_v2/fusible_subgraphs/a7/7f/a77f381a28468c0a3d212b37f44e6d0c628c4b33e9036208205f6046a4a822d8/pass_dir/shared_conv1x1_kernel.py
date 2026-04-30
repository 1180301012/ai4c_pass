import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # High-throughput large M configs (large batches / large spatial)
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=8, num_stages=4),
        # Balanced configs
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        # Small M / small-batch configs (improve SM utilization)
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['N', 'C_in', 'H_out', 'W_out', 'C_out'],
)
@triton.jit
def conv1x1_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    stride_h, stride_w,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    1x1 convolution GEMM kernel (NCHW layout).
    output[n, c_out, h_out, w_out] = sum_k x[n, k, h_out*sh, w_out*sw] * w[c_out, k]
    """
    pid = tl.program_id(0)
    num_m = tl.cdiv(N * H_out * W_out, BLOCK_M)
    pid_m = pid % num_m
    pid_n = pid // num_m

    # Decode (batch, h_out, w_out) from pid_m
    HW_out = H_out * W_out
    bhw_start = pid_m * BLOCK_M
    bhw_offs = bhw_start + tl.arange(0, BLOCK_M)
    bhw_mask = bhw_offs < N * HW_out

    b_idx = bhw_offs // HW_out
    hw_rem = bhw_offs % HW_out
    h_idx = hw_rem // W_out
    w_idx = hw_rem % W_out

    # Absolute input coordinates (accounting for stride)
    h_in = h_idx * stride_h
    w_in = w_idx * stride_w

    c_out_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_out_mask = c_out_offs < C_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop: accumulate over input channels
    for k in range(0, C_in, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        k_mask = k_offs < C_in

        # Load x tile: [BLOCK_M, BLOCK_K]
        # x[b, c, h, w] at b*C_in*H_in*W_in + c*H_in*W_in + h_in*W_in + w_in
        x_ptrs = (x_ptr
                  + b_idx[:, None] * (C_in * H_in * W_in)
                  + k_offs[None, :] * (H_in * W_in)
                  + h_in[:, None] * W_in
                  + w_in[:, None])
        x = tl.load(x_ptrs, mask=bhw_mask[:, None] & k_mask[None, :], other=0.0)

        # Load w tile: [BLOCK_N, BLOCK_K]  (w stored as [C_out, C_in])
        w_ptrs = w_ptr + c_out_offs[:, None] * C_in + k_offs[None, :]
        w = tl.load(w_ptrs, mask=c_out_mask[:, None] & k_mask[None, :], other=0.0)

        # acc += x @ w^T  -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(x, tl.trans(w))

    # Store output in NCHW format: out[b, c_out, h_out, w_out]
    out_ptrs = (out_ptr
                + b_idx[:, None] * (C_out * HW_out)
                + c_out_offs[None, :] * HW_out
                + h_idx[:, None] * W_out
                + w_idx[:, None])
    out_mask = bhw_mask[:, None] & c_out_mask[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


@torch.fx.wrap
def conv1x1_dispatch(in_0, in_1, route):
    """
    Dispatch wrapper for 1x1 conv.
    in_0: weight [C_out, C_in, 1, 1]  (may be on CPU)
    in_1: input  [N, C_in, H_in, W_in] on CUDA
    route: string identifying stride (e.g. "1" or "2")
    Returns: full conv output [N, C_out, H_out, W_out]
    """
    stride_val = int(route)

    C_out = in_0.shape[0]
    C_in = in_0.shape[1]
    N = in_1.shape[0]
    H_in = in_1.shape[2]
    W_in = in_1.shape[3]

    stride_h = stride_val
    stride_w = stride_val
    H_out = (H_in // stride_h)
    W_out = (W_in // stride_w)

    # Move weight to same device as input (handles CPU weights)
    if in_0.device != in_1.device:
        w = in_0.to(in_1.device)
    else:
        w = in_0

    out = torch.empty((N, C_out, H_out, W_out), dtype=in_1.dtype, device=in_1.device)

    grid = lambda META: (triton.cdiv(N * H_out * W_out, META['BLOCK_M']) * triton.cdiv(C_out, META['BLOCK_N']),)

    conv1x1_kernel[grid](
        in_1, w, out,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        stride_h, stride_w,
    )

    return out