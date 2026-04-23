import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    einsum = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    tmp_2 = torch.cat([in_0, einsum], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[(Ellipsis, slice(None, 64, None))]
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_einsum_cat_softmax_kernel(
    in0_ptr, in1_ptr, in2_ptr, out_ptr,
    B, H, W, C, J,
    s_in0_0, s_in0_1, s_in0_2, s_in0_3,
    s_in1_0, s_in1_1, s_in1_2, s_in1_3,
    s_in2_0, s_in2_1, s_in2_2, s_in2_3,
    s_out_0, s_out_1, s_out_2, s_out_3,
    BLOCK_J: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    num_hw = H * W
    b = pid // num_hw
    hw = pid % num_hw
    h = hw // W
    w = hw % W

    j_off = tl.arange(0, BLOCK_J)
    c_off = tl.arange(0, BLOCK_C)

    # Load in0[b, h, w, :J] - first half of softmax input (energy values)
    # in0 shape [B, H, W, J], strides [s_in0_0, s_in0_1, s_in0_2, s_in0_3]
    in0_vals = tl.load(
        in0_ptr + b * s_in0_0 + h * s_in0_1 + w * s_in0_2 + j_off * s_in0_3,
        mask=j_off < J,
        other=float('-inf')
    ).to(tl.float32)

    # Load in2[b, :, h, w] - query vector for einsum (C elements)
    # in2 shape [B, C, H, W], strides [s_in2_0, s_in2_1, s_in2_2, s_in2_3]
    in2_vals = tl.load(
        in2_ptr + b * s_in2_0 + c_off * s_in2_1 + h * s_in2_2 + w * s_in2_3,
        mask=c_off < C,
        other=0.0
    ).to(tl.float32)

    # Load in1[b, :, h, :] as [C, J] - key matrix for einsum
    # in1 shape [B, C, H, J], strides [s_in1_0, s_in1_1, s_in1_2, s_in1_3]
    in1_vals = tl.load(
        in1_ptr + b * s_in1_0 + c_off[:, None] * s_in1_1 + h * s_in1_2 + j_off[None, :] * s_in1_3,
        mask=(c_off[:, None] < C) & (j_off[None, :] < J),
        other=0.0
    ).to(tl.float32)

    # Compute einsum: result[j] = sum_c in2[c] * in1[c, j]
    # This is: for each j, dot product of query and key over channel dimension C
    einsum_vals = tl.sum(in2_vals[:, None] * in1_vals, axis=0)  # [BLOCK_J]

    # Softmax over 2*J = 128 elements (in0 concatenated with einsum along dim -1)
    # Step 1: Find max across both halves
    max_in0 = tl.max(in0_vals, axis=0)
    max_einsum = tl.max(einsum_vals, axis=0)
    max_val = tl.maximum(max_in0, max_einsum)

    # Step 2: Compute exp(x - max) for both halves
    exp_in0 = tl.exp(in0_vals - max_val)
    exp_einsum = tl.exp(einsum_vals - max_val)

    # Step 3: Sum all exponentials
    sum_exp = tl.sum(exp_in0, axis=0) + tl.sum(exp_einsum, axis=0)

    # Step 4: Normalize to get softmax values
    softmax_in0 = exp_in0 / sum_exp
    softmax_einsum = exp_einsum / sum_exp

    # Store full softmax result [B, H, W, 2*J]
    # out shape [B, H, W, 2*J], strides [s_out_0, s_out_1, s_out_2, s_out_3]
    # First half (indices 0:J) = softmax of in0 values
    tl.store(
        out_ptr + b * s_out_0 + h * s_out_1 + w * s_out_2 + j_off * s_out_3,
        softmax_in0,
        mask=j_off < J
    )
    # Second half (indices J:2J) = softmax of einsum values
    tl.store(
        out_ptr + b * s_out_0 + h * s_out_1 + w * s_out_2 + (j_off + J) * s_out_3,
        softmax_einsum,
        mask=j_off < J
    )


@torch.fx.wrap
def fused_einsum_cat_softmax(in_0, in_1, in_2):
    B = in_0.shape[0]
    H = in_0.shape[1]
    W = in_0.shape[2]
    J = in_0.shape[3]  # Last dim size (64)
    C = in_1.shape[1]  # Channel dim size (64)

    BLOCK_J = J  # 64
    BLOCK_C = C  # 64

    # Allocate output tensor [B, H, W, 2*J] = [B, 64, 64, 128]
    out = torch.empty((B, H, W, 2 * J), dtype=in_0.dtype, device=in_0.device)

    grid = (B * H * W,)

    fused_einsum_cat_softmax_kernel[grid](
        in0_ptr=in_0,
        in1_ptr=in_1,
        in2_ptr=in_2,
        out_ptr=out,
        B=B, H=H, W=W, C=C, J=J,
        s_in0_0=in_0.stride()[0], s_in0_1=in_0.stride()[1],
        s_in0_2=in_0.stride()[2], s_in0_3=in_0.stride()[3],
        s_in1_0=in_1.stride()[0], s_in1_1=in_1.stride()[1],
        s_in1_2=in_1.stride()[2], s_in1_3=in_1.stride()[3],
        s_in2_0=in_2.stride()[0], s_in2_1=in_2.stride()[1],
        s_in2_2=in_2.stride()[2], s_in2_3=in_2.stride()[3],
        s_out_0=out.stride()[0], s_out_1=out.stride()[1],
        s_out_2=out.stride()[2], s_out_3=out.stride()[3],
        BLOCK_J=BLOCK_J,
        BLOCK_C=BLOCK_C,
    )

    # The slice output is just a view of the first J elements of the last dimension
    out_slice = out[..., :J]
    return (out, out_slice)


def replacement_func():
    return fused_einsum_cat_softmax