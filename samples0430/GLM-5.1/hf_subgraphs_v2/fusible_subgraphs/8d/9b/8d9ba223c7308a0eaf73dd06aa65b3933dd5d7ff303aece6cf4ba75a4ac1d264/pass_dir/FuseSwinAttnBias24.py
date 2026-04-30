import torch
import triton
import triton.language as tl

# ========== Triton Matmul Kernel ==========
# Computes C = A @ B^T where A is [M, K], B is [N, K], C is [M, N]

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        a = tl.load(
            a_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak,
            mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            b_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn,
            mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N),
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(a, b)

    tl.store(
        c_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn,
        acc,
        mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N),
    )


# ========== Triton Fused Bias + Softmax Kernel ==========
# For each (b, h, i), computes softmax over j of:
#   sigmoid(linear_result[in_0[i, j], h]) * 16 + in_2[b, h, i, j] + 2 * in_3[b, i, j]

@triton.jit
def fused_bias_softmax_kernel(
    linear_result_ptr,
    in_0_ptr,
    in_2_ptr,
    in_3_ptr,
    out_ptr,
    num_heads, batch_size, H, W,
    lr_stride0, lr_stride1,
    in_0_stride0, in_0_stride1,
    in_2_stride0, in_2_stride1, in_2_stride2, in_2_stride3,
    in_3_stride0, in_3_stride1, in_3_stride2,
    out_stride0, out_stride1, out_stride2, out_stride3,
    BLOCK_J: tl.constexpr,
):
    pid = tl.program_id(0)

    total_rows = batch_size * num_heads * H
    if pid >= total_rows:
        return

    b = pid // (num_heads * H)
    remainder = pid % (num_heads * H)
    h = remainder // H
    i = remainder % H

    j_offsets = tl.arange(0, BLOCK_J)
    j_mask = j_offsets < W

    # Load relative position indices for row i
    indices = tl.load(in_0_ptr + i * in_0_stride0 + j_offsets * in_0_stride1, mask=j_mask, other=0)

    # Load linear result values: linear_result[indices, h]
    lr_vals = tl.load(linear_result_ptr + indices * lr_stride0 + h * lr_stride1, mask=j_mask, other=0.0)

    # Sigmoid * 16 (linear_result is float32)
    bias = 16.0 * tl.sigmoid(lr_vals)

    # Load attention scores: in_2[b, h, i, j]
    attn = tl.load(
        in_2_ptr + b * in_2_stride0 + h * in_2_stride1 + i * in_2_stride2 + j_offsets * in_2_stride3,
        mask=j_mask,
        other=0.0,
    ).to(tl.float32)

    # Load mask: in_3[b, i, j] (used twice in the original model)
    mask_vals = tl.load(
        in_3_ptr + b * in_3_stride0 + i * in_3_stride1 + j_offsets * in_3_stride2,
        mask=j_mask,
        other=0.0,
    ).to(tl.float32)

    # Compute x = bias + attn + 2*mask
    x = bias + attn + 2.0 * mask_vals

    # Softmax along j dimension
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    result = exp_x / sum_exp

    # Store result (conversion to output dtype handled by Triton)
    tl.store(
        out_ptr + b * out_stride0 + h * out_stride1 + i * out_stride2 + j_offsets * out_stride3,
        result,
        mask=j_mask,
    )


# ========== Pattern Function (24-head variant) ==========

def pattern(in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
    linear = torch.nn.functional.linear(in_4, in_1, None)
    tmp_3 = linear.view(-1, 24)
    tmp_4 = in_0.view(-1)
    tmp_5 = tmp_3[tmp_4]
    tmp_6 = tmp_5.view(64, 64, -1)
    tmp_7 = tmp_6.permute(2, 0, 1)
    tmp_8 = tmp_7.contiguous()
    tmp_9 = torch.sigmoid(tmp_8)
    tmp_10 = 16 * tmp_9
    tmp_11 = tmp_10.unsqueeze(0)
    tmp_12 = in_2 + tmp_11
    tmp_13 = tmp_12.view(1, 16, 24, 64, 64)
    tmp_14 = in_3.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_16 = tmp_13 + tmp_15
    tmp_17 = in_3.unsqueeze(1)
    tmp_18 = tmp_17.unsqueeze(0)
    tmp_19 = tmp_16 + tmp_18
    tmp_20 = tmp_19.view(-1, 24, 64, 64)
    tmp_21 = torch.nn.functional.softmax(tmp_20, dim = -1)
    tmp_22 = torch.nn.functional.dropout(tmp_21, 0.0, False, False)
    return (tmp_22,)


# ========== Replacement Args ==========

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4, "route_24")


# ========== Shared Dispatch Wrapper ==========

@torch.fx.wrap
def fused_swin_attn_dispatch(in_0, in_1, in_2, in_3, in_4, route):
    if route == "route_12":
        num_heads = 12
        batch_size = 64
    elif route == "route_24":
        num_heads = 24
        batch_size = 16
    else:
        raise ValueError(f"Unknown route: {route}")

    H = 64
    W = 64
    M = in_4.shape[1] * in_4.shape[2]  # 15 * 15 = 225
    K = in_4.shape[3]  # 512

    dtype = in_2.dtype
    device = in_4.device

    # Step 1: Compute linear result via Triton matmul
    # in_4: [1, 15, 15, 512] viewed as [225, 512]
    # in_1: [num_heads, 512]
    # linear_result: [225, num_heads] in float32 for numerical stability

    linear_result = torch.empty((M, num_heads), dtype=torch.float32, device=device)

    BLOCK_M_MM = 32
    BLOCK_N_MM = 32
    BLOCK_K_MM = 32

    num_m_blocks = (M + BLOCK_M_MM - 1) // BLOCK_M_MM
    num_n_blocks = (num_heads + BLOCK_N_MM - 1) // BLOCK_N_MM

    # Effective 2D strides for in_4 viewed as [225, 512]
    # For contiguous [1, 15, 15, 512]: stride_am = stride(2) = 512, stride_ak = stride(3) = 1
    stride_am = in_4.stride(2)
    stride_ak = in_4.stride(3)

    matmul_kernel[(num_m_blocks, num_n_blocks)](
        a_ptr=in_4,
        b_ptr=in_1,
        c_ptr=linear_result,
        M=M,
        N=num_heads,
        K=K,
        stride_am=stride_am,
        stride_ak=stride_ak,
        stride_bn=in_1.stride(0),
        stride_bk=in_1.stride(1),
        stride_cm=linear_result.stride(0),
        stride_cn=linear_result.stride(1),
        BLOCK_M=BLOCK_M_MM,
        BLOCK_N=BLOCK_N_MM,
        BLOCK_K=BLOCK_K_MM,
    )

    # Step 2: Compute fused bias + softmax
    # Output shape: [batch_size, num_heads, 64, 64]
    out = torch.empty((batch_size, num_heads, H, W), dtype=dtype, device=device)

    total_programs = batch_size * num_heads * H
    BLOCK_J = 64

    fused_bias_softmax_kernel[(total_programs,)](
        linear_result_ptr=linear_result,
        in_0_ptr=in_0,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        out_ptr=out,
        num_heads=num_heads,
        batch_size=batch_size,
        H=H,
        W=W,
        lr_stride0=linear_result.stride(0),
        lr_stride1=linear_result.stride(1),
        in_0_stride0=in_0.stride(0),
        in_0_stride1=in_0.stride(1),
        in_2_stride0=in_2.stride(0),
        in_2_stride1=in_2.stride(1),
        in_2_stride2=in_2.stride(2),
        in_2_stride3=in_2.stride(3),
        in_3_stride0=in_3.stride(0),
        in_3_stride1=in_3.stride(1),
        in_3_stride2=in_3.stride(2),
        out_stride0=out.stride(0),
        out_stride1=out.stride(1),
        out_stride2=out.stride(2),
        out_stride3=out.stride(3),
        BLOCK_J=BLOCK_J,
    )

    return (out,)


# ========== Replacement Function ==========

def replacement_func():
    return fused_swin_attn_dispatch