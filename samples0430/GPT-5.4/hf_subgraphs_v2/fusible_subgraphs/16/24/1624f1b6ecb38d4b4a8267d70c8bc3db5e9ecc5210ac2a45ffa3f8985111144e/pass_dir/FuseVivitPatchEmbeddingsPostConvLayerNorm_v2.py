import torch
import triton
import triton.language as tl


NUM_PATCH_K = 3 * 2 * 16 * 16
PATCH_T = 2
PATCH_H = 16
PATCH_W = 16


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv3d = torch.conv3d(in_6, in_1, in_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    tmp_7 = conv3d.flatten(2)
    tmp_8 = tmp_7.transpose(1, 2)
    tmp_9 = in_2.tile([1, 1, 1])
    tmp_10 = torch.cat((tmp_9, tmp_8), dim=1)
    tmp_11 = tmp_10 + in_3
    tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (768,), in_5, in_4, 1e-06)
    return (tmp_12, tmp_13)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


@triton.jit
def extract_video_patches_kernel(
    x_ptr,
    patches_ptr,
    x_s0,
    x_s1,
    x_s2,
    x_s3,
    x_s4,
    p_s0,
    p_s1,
    PT,
    PH,
    PW,
    K,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    num_patches = PT * PH * PW
    b = row // num_patches
    patch = row - b * num_patches

    pt = patch // (PH * PW)
    rem = patch - pt * (PH * PW)
    ph = rem // PW
    pw = rem - ph * PW

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    c = offs_k // (PATCH_T * PATCH_H * PATCH_W)
    rem_k0 = offs_k - c * (PATCH_T * PATCH_H * PATCH_W)
    kt = rem_k0 // (PATCH_H * PATCH_W)
    rem_k1 = rem_k0 - kt * (PATCH_H * PATCH_W)
    kh = rem_k1 // PATCH_W
    kw = rem_k1 - kh * PATCH_W

    x_ptrs = (
        x_ptr
        + b * x_s0
        + c * x_s1
        + (pt * PATCH_T + kt) * x_s2
        + (ph * PATCH_H + kh) * x_s3
        + (pw * PATCH_W + kw) * x_s4
    )
    vals = tl.load(x_ptrs, mask=mask_k, other=0.0)

    patch_ptrs = patches_ptr + row * p_s0 + offs_k * p_s1
    tl.store(patch_ptrs, vals, mask=mask_k)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def patch_gemm_with_bias_pos_kernel(
    a_ptr,
    w_ptr,
    bias_ptr,
    pos_ptr,
    out_ptr,
    a_s0,
    a_s1,
    w_s0,
    w_s1,
    w_s2,
    w_s3,
    w_s4,
    pos_s1,
    pos_s2,
    out_s0,
    out_s1,
    out_s2,
    NUM_PATCHES,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * a_s0 + k[None, :] * a_s1
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)

        c = k // (PATCH_T * PATCH_H * PATCH_W)
        rem_k0 = k - c * (PATCH_T * PATCH_H * PATCH_W)
        kt = rem_k0 // (PATCH_H * PATCH_W)
        rem_k1 = rem_k0 - kt * (PATCH_H * PATCH_W)
        kh = rem_k1 // PATCH_W
        kw = rem_k1 - kh * PATCH_W

        w_ptrs = (
            w_ptr
            + offs_n[None, :] * w_s0
            + c[:, None] * w_s1
            + kt[:, None] * w_s2
            + kh[:, None] * w_s3
            + kw[:, None] * w_s4
        )
        w = tl.load(w_ptrs, mask=(k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, w, acc)

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    patch_ids = offs_m % NUM_PATCHES
    pos = tl.load(
        pos_ptr + (patch_ids[:, None] + 1) * pos_s1 + offs_n[None, :] * pos_s2,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0,
    ).to(tl.float32)
    out = acc + bias[None, :] + pos

    batch_ids = offs_m // NUM_PATCHES
    out_ptrs = out_ptr + batch_ids[:, None] * out_s0 + (patch_ids[:, None] + 1) * out_s1 + offs_n[None, :] * out_s2
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=4),
    ],
    key=["H"],
)
@triton.jit
def cls_row_and_layer_norm_kernel(
    cls_ptr,
    pos_ptr,
    gamma_ptr,
    beta_ptr,
    out12_ptr,
    out13_ptr,
    cls_s0,
    cls_s2,
    pos_s1,
    pos_s2,
    out12_s0,
    out12_s1,
    out12_s2,
    out13_s0,
    out13_s1,
    out13_s2,
    SEQ,
    H,
    eps,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // SEQ
    token = row - b * SEQ
    read_token = tl.where(token == 0, 1, token)

    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    cls_vals = tl.load(cls_ptr + b * cls_s0 + offs * cls_s2, mask=mask, other=0.0)
    pos_vals = tl.load(pos_ptr + token * pos_s1 + offs * pos_s2, mask=mask, other=0.0)
    base_vals = tl.load(out12_ptr + b * out12_s0 + read_token * out12_s1 + offs * out12_s2, mask=mask, other=0.0)
    x = tl.where(token == 0, cls_vals + pos_vals, base_vals)

    tl.store(out12_ptr + b * out12_s0 + token * out12_s1 + offs * out12_s2, x, mask=mask)

    x_fp32 = x.to(tl.float32)
    mean = tl.sum(x_fp32, axis=0) / H
    centered = x_fp32 - mean
    var = tl.sum(centered * centered, axis=0) / H
    inv_std = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = centered * inv_std * gamma + beta
    tl.store(out13_ptr + b * out13_s0 + token * out13_s1 + offs * out13_s2, y, mask=mask)


@torch.fx.wrap
def fused_vivit_patch_embed_impl(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    batch = in_6.shape[0]
    out_channels = in_1.shape[0]
    pt = in_6.shape[2] // PATCH_T
    ph = in_6.shape[3] // PATCH_H
    pw = in_6.shape[4] // PATCH_W
    num_patches = pt * ph * pw
    seq = num_patches + 1
    m = batch * num_patches
    k = NUM_PATCH_K

    patches = torch.empty((m, k), device=in_6.device, dtype=in_6.dtype)
    out12 = torch.empty((batch, seq, out_channels), device=in_6.device, dtype=in_6.dtype)
    out13 = torch.empty((batch, seq, out_channels), device=in_6.device, dtype=in_6.dtype)

    extract_video_patches_kernel[(m,)](
        in_6,
        patches,
        in_6.stride(0),
        in_6.stride(1),
        in_6.stride(2),
        in_6.stride(3),
        in_6.stride(4),
        patches.stride(0),
        patches.stride(1),
        pt,
        ph,
        pw,
        k,
        BLOCK_K=2048,
    )

    gemm_grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(out_channels, META["BLOCK_N"]),)
    patch_gemm_with_bias_pos_kernel[gemm_grid](
        patches,
        in_1,
        in_0,
        in_3,
        out12,
        patches.stride(0),
        patches.stride(1),
        in_1.stride(0),
        in_1.stride(1),
        in_1.stride(2),
        in_1.stride(3),
        in_1.stride(4),
        in_3.stride(1),
        in_3.stride(2),
        out12.stride(0),
        out12.stride(1),
        out12.stride(2),
        num_patches,
        m,
        out_channels,
        k,
    )

    cls_row_and_layer_norm_kernel[(batch * seq,)](
        in_2,
        in_3,
        in_5,
        in_4,
        out12,
        out13,
        in_2.stride(0),
        in_2.stride(2),
        in_3.stride(1),
        in_3.stride(2),
        out12.stride(0),
        out12.stride(1),
        out12.stride(2),
        out13.stride(0),
        out13.stride(1),
        out13.stride(2),
        seq,
        out_channels,
        1e-6,
    )

    return out12, out13


def fused_vivit_patch_embed(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    outs = fused_vivit_patch_embed_impl(in_0, in_1, in_2, in_3, in_4, in_5, in_6)
    return outs[0], outs[1]


def replacement_func():
    return fused_vivit_patch_embed