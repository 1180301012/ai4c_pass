import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.permute(0, 3, 1, 2)
    return tmp_3


def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)


@triton.jit
def fused_linear_permute_kernel(
    in_3_ptr, in_1_ptr, in_0_ptr, out_ptr,
    B, C_out, H, W,
    stride_in3_b, stride_in3_h, stride_in3_w, stride_in3_c,
    stride_wt_out, stride_wt_in,
    stride_out_b, stride_out_h, stride_out_w, stride_out_c,
    BLOCK_M: tl.constexpr,
    GROUP_C: tl.constexpr,
    C_IN: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)

    b = pid_b

    m_off = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_off < H * W

    i = m_off // W
    j = m_off % W

    n_off = tl.arange(0, GROUP_C)

    # Load bias [GROUP_C]
    bias = tl.load(in_0_ptr + n_off).to(tl.float32)

    # Initialize accumulator [BLOCK_M, GROUP_C] with bias
    acc = bias[None, :] + tl.zeros([BLOCK_M, GROUP_C], dtype=tl.float32)

    # Outer product accumulation over C_IN (unrolled for C_IN=3)
    # Pre-load all weight rows for efficiency
    wt0 = tl.load(in_1_ptr + n_off * stride_wt_out + 0 * stride_wt_in).to(tl.float32)
    wt1 = tl.load(in_1_ptr + n_off * stride_wt_out + 1 * stride_wt_in).to(tl.float32)
    wt2 = tl.load(in_1_ptr + n_off * stride_wt_out + 2 * stride_wt_in).to(tl.float32)

    # Load input for each channel - these are 1D vector loads
    in0 = tl.load(in_3_ptr + b * stride_in3_b + i * stride_in3_h + j * stride_in3_w + 0 * stride_in3_c, mask=m_mask, other=0.0).to(tl.float32)
    in1 = tl.load(in_3_ptr + b * stride_in3_b + i * stride_in3_h + j * stride_in3_w + 1 * stride_in3_c, mask=m_mask, other=0.0).to(tl.float32)
    in2 = tl.load(in_3_ptr + b * stride_in3_b + i * stride_in3_h + j * stride_in3_w + 2 * stride_in3_c, mask=m_mask, other=0.0).to(tl.float32)

    # Compute all outputs at once: [BLOCK_M, GROUP_C]
    acc += in0[:, None] * wt0[None, :]
    acc += in1[:, None] * wt1[None, :]
    acc += in2[:, None] * wt2[None, :]

    # Store output in original layout: [B, H, W, C_out] (better write coalescing)
    out_offsets = b * stride_out_b + i[:, None] * stride_out_h + j[:, None] * stride_out_w + n_off[None, :] * stride_out_c
    tl.store(out_ptr + out_offsets, acc, mask=m_mask[:, None])


@torch.fx.wrap
def fused_linear_permute(in_0, in_1, in_3):
    # Ensure weights are on the same device as input
    device = in_3.device
    if in_0.device != device:
        in_0 = in_0.to(device)
    if in_1.device != device:
        in_1 = in_1.to(device)

    B, H, W, C_in = in_3.shape
    C_out = in_1.shape[0]

    # Output in original layout: [B, H, W, C_out] for better write coalescing
    out = torch.empty((B, H, W, C_out), dtype=in_3.dtype, device=device)

    BLOCK_M = 128  # Fixed block size for good occupancy
    GROUP_C = C_out  # 16, power of 2
    C_IN = C_in      # 3

    num_m_blocks = triton.cdiv(H * W, BLOCK_M)
    grid = (num_m_blocks, B)

    fused_linear_permute_kernel[grid](
        in_3_ptr=in_3, in_1_ptr=in_1, in_0_ptr=in_0, out_ptr=out,
        B=B, C_out=C_out, H=H, W=W,
        stride_in3_b=in_3.stride(0), stride_in3_h=in_3.stride(1),
        stride_in3_w=in_3.stride(2), stride_in3_c=in_3.stride(3),
        stride_wt_out=in_1.stride(0), stride_wt_in=in_1.stride(1),
        stride_out_b=out.stride(0), stride_out_h=out.stride(1),
        stride_out_w=out.stride(2), stride_out_c=out.stride(3),
        BLOCK_M=BLOCK_M,
        GROUP_C=GROUP_C,
        C_IN=C_IN,
    )

    # Apply permute to get [B, C_out, H, W] - this is a view (no data copy)
    return out.permute(0, 3, 1, 2)


def replacement_func():
    return fused_linear_permute