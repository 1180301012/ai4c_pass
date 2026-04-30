import torch
import triton
import triton.language as tl
from pass_dir.build_pattern import build_pattern_gm


pattern = build_pattern_gm()


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['B', 'C', 'H', 'W', 'J'],
)
@triton.jit
def fused_einsum_scale_add_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, in_4_ptr, out_ptr,
    B, C, H, W, J,
    stride_in4_b, stride_in4_c, stride_in4_h, stride_in4_j,
    stride_in1_b, stride_in1_h, stride_in1_w, stride_in1_j,
    stride_in3_b, stride_in3_c, stride_in3_h, stride_in3_w,
    stride_in2_b, stride_in2_c, stride_in2_h, stride_in2_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (num_m_blocks, B * H)
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # C dimension
    offs_n = tl.arange(0, BLOCK_N)  # W dimension

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop (matmul accumulation over J dimension)
    for k_start in range(0, J, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load A: in_4[b, offs_m, h, offs_k] -> shape [BLOCK_M, BLOCK_K]
        a_ptrs = (in_4_ptr + b * stride_in4_b + h * stride_in4_h
                  + offs_m[:, None] * stride_in4_c + offs_k[None, :] * stride_in4_j)
        a_mask = (offs_m[:, None] < C) & (offs_k[None, :] < J)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B^T: bt[k, n] = in_1[b, h, n, k] -> shape [BLOCK_K, BLOCK_N]
        bt_ptrs = (in_1_ptr + b * stride_in1_b + h * stride_in1_h
                   + offs_k[:, None] * stride_in1_j + offs_n[None, :] * stride_in1_w)
        bt_mask = (offs_k[:, None] < J) & (offs_n[None, :] < W)
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0)

        # Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, bt)

    # Epilogue: result = (einsum + in_3) * scale + in_2
    out_mask = (offs_m[:, None] < C) & (offs_n[None, :] < W)

    # Load in_3[b, offs_m, h, offs_n]
    in3_ptrs = (in_3_ptr + b * stride_in3_b + h * stride_in3_h
                + offs_m[:, None] * stride_in3_c + offs_n[None, :] * stride_in3_w)
    in3 = tl.load(in3_ptrs, mask=out_mask, other=0.0).to(tl.float32)

    # Load scalar scale
    scale = tl.load(in_0_ptr).to(tl.float32)

    # Load in_2[b, offs_m, h, offs_n]
    in2_ptrs = (in_2_ptr + b * stride_in2_b + h * stride_in2_h
                + offs_m[:, None] * stride_in2_c + offs_n[None, :] * stride_in2_w)
    in2 = tl.load(in2_ptrs, mask=out_mask, other=0.0).to(tl.float32)

    # Compute fused result: (einsum + in_3) * scale + in_2
    result = (acc + in3) * scale + in2

    # Store output
    out_ptrs = (out_ptr + b * stride_out_b + h * stride_out_h
                + offs_m[:, None] * stride_out_c + offs_n[None, :] * stride_out_w)
    tl.store(out_ptrs, result.to(out_ptr.dtype.element_ty), mask=out_mask)


@torch.fx.wrap
def fused_einsum_scale_add(in_0, in_1, in_2, in_3, in_4):
    # in_0: scalar []
    # in_1: [B, H, W, J]
    # in_2: [B, C, H, W]
    # in_3: [B, C, H, W]
    # in_4: [B, C, H, J]
    B = in_4.shape[0]
    C = in_4.shape[1]
    H = in_4.shape[2]
    J = in_4.shape[3]
    W = in_1.shape[2]

    out = torch.empty(B, C, H, W, device=in_4.device, dtype=in_4.dtype)

    def grid(META):
        return (triton.cdiv(C, META['BLOCK_M']), B * H)

    fused_einsum_scale_add_kernel[grid](
        in_0, in_1, in_2, in_3, in_4, out,
        B, C, H, W, J,
        in_4.stride(0), in_4.stride(1), in_4.stride(2), in_4.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )

    return out


def replacement_func():
    return fused_einsum_scale_add