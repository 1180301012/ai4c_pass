import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'K', 'N_out'],
)
@triton.jit
def _fused_1x1conv_slice_kernel(
    w_ptr,          # weight tensor [N_out, K]  (row-major)
    x_ptr,          # input  tensor [B, K, H_in, W_in]  (row-major)
    out_ptr,        # output tensor [B, N_out, H_out, W_out]  (row-major)
    B, H_in, W_in,
    H_out, W_out,
    K, N_out,
    M,
    stride_h, stride_w,
    K_out,   # FIRST K output channels we compute (slice)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Fused 1x1 conv + slice kernel.
    Computes out[:, :K_out, :, :] without materialising all N_out channels.
    For stride=1: stride_h=1, stride_w=1  (no im2col, just GEMM)
    For stride=2: stride_h=2, stride_w=2  (im2col w/ stride 2)
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(K_out, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Row indices: which (batch, h_out, w_out) positions this block handles
    m_start = pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    m_mask = m_offs < M                          # [BLOCK_M]

    # Column indices: first K_out output channels
    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    n_mask = n_offs < K_out                      # [BLOCK_N]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Map m -> (batch, h_out, w_out)
    HW_out = H_out * W_out
    b_idx  = m_offs // HW_out          # batch index  [BLOCK_M]
    hw_idx = m_offs % HW_out
    h_idx  = hw_idx // W_out           # h_out index  [BLOCK_M]
    w_idx  = hw_idx % W_out            # w_out index  [BLOCK_M]

    # Accumulate over K (input channels) in BLOCK_K chunks
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        acc += tl.dot(
            tl.load(
                w_ptr  + n_offs[:, None] * K + (k * BLOCK_K + tl.arange(0, BLOCK_K))[None, :],
                mask=(n_mask[:, None] & ((k * BLOCK_K + tl.arange(0, BLOCK_K))[None, :] < K)),
                other=0.0,
            ),
            tl.load(
                x_ptr + b_idx[:, None] * (K * H_in * W_in)
                + (k * BLOCK_K + tl.arange(0, BLOCK_K))[None, :] * (H_in * W_in)
                + (h_idx[:, None] * W_in + w_idx[:, None]),
                mask=m_mask[:, None] & ((k * BLOCK_K + tl.arange(0, BLOCK_K))[None, :] < K),
                other=0.0,
            ),
        )

    # Compute flat output index:  out[b, k_out, h_out, w_out]
    out_offs = b_idx[:, None] * (N_out * HW_out) + n_offs[None, :] * HW_out + h_idx[:, None] * W_out + w_idx[:, None]
    tl.store(
        out_ptr + out_offs,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@torch.fx.wrap
def conv1x1_slice_dispatch(x, w, route):
    """
    Dispatch wrapper for 1x1 conv with channel slice.

    x   : input tensor  [B, C_in,  H_in,  W_in ]
    w   : weight tensor [C_out, C_in, 1, 1 ]  (or [C_out, C_in])
    route: string encoding (stride, K_out, return_order)

    Return order:
      "s2r0"  → (full_conv, sliced)   (stride=2,   tmp_2, conv2d)
      "s1r0"  → (full_conv, sliced)   (stride=1,   tmp_2, conv2d)
      "s2r1"  → (sliced,    full_conv) (stride=2,   conv2d, tmp_2)
      "s1r1"  → (sliced,    full_conv) (stride=1,   conv2d, tmp_2)
    """
    if route == "s2_2048r0":
        K_out_val, OH_val, OW_val = 2048, 2, 2
    elif route == "s1_2048r0":
        K_out_val, OH_val, OW_val = 2048, 1, 1
    elif route == "s1_2048r1":
        K_out_val, OH_val, OW_val = 2048, 1, 1
    elif route == "s1_64r0":
        K_out_val, OH_val, OW_val = 64, 1, 1
    elif route == "s1_64r1":
        K_out_val, OH_val, OW_val = 64, 1, 1
    elif route == "s1_512r0":
        K_out_val, OH_val, OW_val = 512, 1, 1
    elif route == "s1_512r1":
        K_out_val, OH_val, OW_val = 512, 1, 1
    elif route == "s1_128r0":
        K_out_val, OH_val, OW_val = 128, 1, 1
    elif route == "s1_128r1":
        K_out_val, OH_val, OW_val = 128, 1, 1
    elif route == "s1_256r0":
        K_out_val, OH_val, OW_val = 256, 1, 1
    elif route == "s1_256r1":
        K_out_val, OH_val, OW_val = 256, 1, 1
    elif route == "s1_1024r0":
        K_out_val, OH_val, OW_val = 1024, 1, 1
    elif route == "s1_1024r1":
        K_out_val, OH_val, OW_val = 1024, 1, 1
    elif route == "s2_1024r0":
        K_out_val, OH_val, OW_val = 1024, 2, 2
    elif route == "s2_512r0":
        K_out_val, OH_val, OW_val = 512, 2, 2
    elif route == "s2_256r1":
        K_out_val, OH_val, OW_val = 256, 2, 2
    elif route == "s2_64r1":
        K_out_val, OH_val, OW_val = 64, 2, 2
    else:
        K_out_val, OH_val, OW_val = 2048, 2, 2

    device = x.device
    x_4d = x if x.dim() == 4 else x.view(-1) if x.numel() == 1 else x

    stride_h, stride_w = (OH_val, OW_val) if OH_val == 1 else (OH_val * 2, OW_val * 2)

    w_2d = w.view(-1, w.shape[-1]) if w.dim() == 4 else w.view(-1, w.numel() // w.shape[0])
    B, K, H_in, W_in = x_4d.shape
    N_out = w_2d.shape[0]
    M = B * H_in * W_in

    H_out = H_in // stride_h
    W_out = W_in // stride_w

    conv_out = torch.empty((B, N_out, H_out, W_out), dtype=x.dtype, device=device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(K_out_val, meta['BLOCK_N']),)

    _fused_1x1conv_slice_kernel[grid](
        w_2d, x_4d, conv_out,
        B, H_in, W_in,
        H_out, W_out,
        K, N_out, M,
        stride_h, stride_w,
        K_out_val,
    )

    full_conv = conv_out.view(B, N_out, H_out, W_out)
    slice_out = full_conv[:, :K_out_val, :, :].contiguous()
    full_view = full_conv.contiguous()

    if route in ("s2_2048r0", "s1_2048r0", "s2_2048r1", "s1_2048r1"):
        return slice_out, full_view
    elif route in ("s1_64r0", "s1_64r1"):
        return slice_out, full_view
    elif route in ("s2_1024r0", "s2_512r0"):
        return slice_out, full_view
    elif route in ("s1_512r0", "s1_512r1"):
        return slice_out, full_view
    elif route in ("s1_128r0", "s1_128r1"):
        return slice_out, full_view
    elif route in ("s1_256r0", "s1_256r1"):
        return slice_out, full_view
    elif route in ("s1_1024r0", "s1_1024r1"):
        return slice_out, full_view
    elif route in ("s2_256r1", "s2_64r1"):
        return slice_out, full_view
    else:
        return slice_out, full_view