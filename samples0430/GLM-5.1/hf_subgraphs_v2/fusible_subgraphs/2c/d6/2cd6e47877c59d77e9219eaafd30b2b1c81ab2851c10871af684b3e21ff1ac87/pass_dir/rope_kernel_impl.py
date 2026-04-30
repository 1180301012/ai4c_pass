import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def rope_fused_kernel(
    out1_ptr, out2_ptr,
    in0_ptr, in1_ptr, in2_ptr, in3_ptr, in4_ptr, in5_ptr,
    # Output strides (for shape [1, H, S1, D])
    out1_s0, out1_s1, out1_s2, out1_s3,
    out2_s0, out2_s1, out2_s2, out2_s3,
    # Input strides
    in0_s0, in0_s1,  # [S, 2*D]
    in1_s0, in1_s1,  # [S, D]
    in2_s0, in2_s1, in2_s2, in2_s3,  # [1, H, 1, D]
    in3_s0, in3_s1, in3_s2, in3_s3,  # [1, H, S, D]
    in4_s0, in4_s1, in4_s2, in4_s3,  # [1, H, S1, D]
    in5_s0, in5_s1,  # [S, D]
    # Dimensions
    H, S1, D, S, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decode flat index to (h, s_out, d) in output shape [1, H, S1, D]
    # batch dim b is always 0 (shape[0] = 1)
    d = offsets % D
    s_out = (offsets // D) % S1
    h = (offsets // (D * S1)) % H

    is_first = s_out == 0
    not_first = ~is_first
    # Safe s_rest: clamp to 0 when is_first (value is unused due to mask)
    s_rest = tl.where(is_first, tl.zeros_like(s_out), s_out - 1)
    d_even = (d % 2) == 0

    # ---- Part 1: q output (tmp_10) ----
    out1_offset = h * out1_s1 + s_out * out1_s2 + d * out1_s3

    # First token: load from in_2 [1, H, 1, D]
    in2_offset = h * in2_s1 + d * in2_s3
    first_q = tl.load(in2_ptr + in2_offset, mask=mask & is_first, other=0.0)

    # Rotated: load from in_3 [1, H, S, D]
    in3_offset = h * in3_s1 + s_rest * in3_s2 + d * in3_s3
    x_q = tl.load(in3_ptr + in3_offset, mask=mask & not_first, other=0.0)

    # Rotate half for q: even d -> -x[d+1], odd d -> x[d-1]
    in3_offset_next = h * in3_s1 + s_rest * in3_s2 + (d + 1) * in3_s3
    in3_offset_prev = h * in3_s1 + s_rest * in3_s2 + (d - 1) * in3_s3
    x_q_next = tl.load(in3_ptr + in3_offset_next, mask=mask & not_first & (d < D - 1), other=0.0)
    x_q_prev = tl.load(in3_ptr + in3_offset_prev, mask=mask & not_first & (d > 0), other=0.0)
    rotate_q = tl.where(d_even, -x_q_next, x_q_prev)

    # Load embeddings
    in1_offset = s_rest * in1_s0 + d * in1_s1  # cos_emb[s_rest, d]
    in5_offset = s_rest * in5_s0 + d * in5_s1  # sin_emb[s_rest, d]
    cos_q = tl.load(in1_ptr + in1_offset, mask=mask & not_first, other=0.0)
    sin_q = tl.load(in5_ptr + in5_offset, mask=mask & not_first, other=0.0)

    # Compute q output: x * cos + rotate(x) * sin
    rotated_q = x_q * cos_q + rotate_q * sin_q
    out1_val = tl.where(is_first, first_q, rotated_q)

    tl.store(out1_ptr + out1_offset, out1_val, mask=mask)

    # ---- Part 2: k output (tmp_25) ----
    out2_offset = h * out2_s1 + s_out * out2_s2 + d * out2_s3

    # First token: load from in_4 at s=0 [1, H, S1, D]
    in4_first_offset = h * in4_s1 + d * in4_s3  # in_4[b, h, 0, d] where s=0
    first_k = tl.load(in4_ptr + in4_first_offset, mask=mask & is_first, other=0.0)

    # Rotated: load from in_4 at s=s_out [1, H, S1, D]
    in4_offset = h * in4_s1 + s_out * in4_s2 + d * in4_s3
    x_k = tl.load(in4_ptr + in4_offset, mask=mask & not_first, other=0.0)

    # Rotate half for k
    in4_offset_next = h * in4_s1 + s_out * in4_s2 + (d + 1) * in4_s3
    in4_offset_prev = h * in4_s1 + s_out * in4_s2 + (d - 1) * in4_s3
    x_k_next = tl.load(in4_ptr + in4_offset_next, mask=mask & not_first & (d < D - 1), other=0.0)
    x_k_prev = tl.load(in4_ptr + in4_offset_prev, mask=mask & not_first & (d > 0), other=0.0)
    rotate_k = tl.where(d_even, -x_k_next, x_k_prev)

    # Load pos_embed [S, 2*D]
    # First half: in_0[s_rest, d], Second half: in_0[s_rest, d + D]
    in0_offset_first = s_rest * in0_s0 + d * in0_s1
    in0_offset_second = s_rest * in0_s0 + (d + D) * in0_s1
    pos_first = tl.load(in0_ptr + in0_offset_first, mask=mask & not_first, other=0.0)
    pos_second = tl.load(in0_ptr + in0_offset_second, mask=mask & not_first, other=0.0)

    # Compute k output: x * pos_second + rotate(x) * pos_first
    # This matches: k_rest * tensor_split[1] + rotate(k_rest) * tensor_split[0]
    rotated_k = x_k * pos_second + rotate_k * pos_first
    out2_val = tl.where(is_first, first_k, rotated_k)

    tl.store(out2_ptr + out2_offset, out2_val, mask=mask)


@torch.fx.wrap
def rope_dispatch_wrapper(in_0, in_1, in_2, in_3, in_4, in_5, in_6, route):
    # Extract dimensions from input shapes
    H = in_3.shape[1]
    S = in_3.shape[2]
    D = in_3.shape[3]
    S1 = S + 1

    dtype = in_6.dtype
    device = in_3.device

    # Allocate output tensors
    # out1 corresponds to tmp_10 (q output)
    # out2 corresponds to tmp_25 (k output)
    out1 = torch.empty((1, H, S1, D), dtype=dtype, device=device)
    out2 = torch.empty((1, H, S1, D), dtype=dtype, device=device)

    n_elements = H * S1 * D

    grid = ((n_elements + 256 - 1) // 256,)  # Will be overridden by autotune

    rope_fused_kernel[grid](
        out1, out2,
        in_0, in_1, in_2, in_3, in_4, in_5,
        out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
        in_0.stride(0), in_0.stride(1),
        in_1.stride(0), in_1.stride(1),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        in_4.stride(0), in_4.stride(1), in_4.stride(2), in_4.stride(3),
        in_5.stride(0), in_5.stride(1),
        H, S1, D, S, n_elements,
    )

    # Return order matches model: (tmp_25, tmp_10) = (k_output, q_output)
    return (out2, out1)