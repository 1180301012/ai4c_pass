import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32, 'BLOCK_C': 16}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64, 'BLOCK_C': 16}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64, 'BLOCK_C': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64, 'BLOCK_C': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_C': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'BLOCK_C': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'BLOCK_C': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256, 'BLOCK_C': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512, 'BLOCK_C': 16}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512, 'BLOCK_C': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512, 'BLOCK_C': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024, 'BLOCK_C': 16}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024, 'BLOCK_C': 32}, num_warps=16),
    ],
    key=['B', 'C', 'H', 'W'],
)
@triton.jit
def fused_mul_sum_sigmoid_kernel(
    in0_ptr, in1_ptr, out_ptr,
    B, C, H, W,
    stride_in0_0, stride_in0_1, stride_in0_2, stride_in0_3,
    stride_in1_0, stride_in1_1, stride_in1_2, stride_in1_3,
    stride_out_0, stride_out_1, stride_out_2, stride_out_3,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    total_spatial = B * H * W
    mask = offsets < total_spatial

    # Decompose offset into (b, h, w)
    HW = H * W
    b_idx = offsets // HW
    hw_idx = offsets % HW
    h_idx = hw_idx // W
    w_idx = hw_idx % W

    # Accumulate in float32 for precision
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C

        # Compute input pointers: in0[b, c, h, w]
        in0_ptrs = in0_ptr + (b_idx[:, None] * stride_in0_0 +
                              c_offsets[None, :] * stride_in0_1 +
                              h_idx[:, None] * stride_in0_2 +
                              w_idx[:, None] * stride_in0_3)

        in1_ptrs = in1_ptr + (b_idx[:, None] * stride_in1_0 +
                              c_offsets[None, :] * stride_in1_1 +
                              h_idx[:, None] * stride_in1_2 +
                              w_idx[:, None] * stride_in1_3)

        combined_mask = mask[:, None] & c_mask[None, :]

        # Load and upcast to float32 for accurate accumulation
        in0_vals = tl.load(in0_ptrs, mask=combined_mask, other=0.0).to(tl.float32)
        in1_vals = tl.load(in1_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        # Multiply and accumulate across channel dimension
        product = in0_vals * in1_vals
        acc += tl.sum(product, axis=1)

    # Apply sigmoid
    result = tl.sigmoid(acc)

    # Store result at output[b, 0, h, w] (dim 1 has size 1, so index is always 0)
    out_ptrs = out_ptr + (b_idx * stride_out_0 +
                          h_idx * stride_out_2 +
                          w_idx * stride_out_3)
    tl.store(out_ptrs, result, mask=mask)


@torch.fx.wrap
def fused_mul_sum_sigmoid(in_0, in_1):
    B, C, H, W = in_0.shape
    out_dtype = in_0.dtype

    # Create output tensor of shape [B, 1, H, W]
    out = torch.empty((B, 1, H, W), dtype=out_dtype, device=in_0.device)

    total_spatial = B * H * W

    grid = lambda meta: ((total_spatial + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_mul_sum_sigmoid_kernel[grid](
        in0_ptr=in_0, in1_ptr=in_1, out_ptr=out,
        B=B, C=C, H=H, W=W,
        stride_in0_0=in_0.stride()[0], stride_in0_1=in_0.stride()[1],
        stride_in0_2=in_0.stride()[2], stride_in0_3=in_0.stride()[3],
        stride_in1_0=in_1.stride()[0], stride_in1_1=in_1.stride()[1],
        stride_in1_2=in_1.stride()[2], stride_in1_3=in_1.stride()[3],
        stride_out_0=out.stride()[0], stride_out_1=out.stride()[1],
        stride_out_2=out.stride()[2], stride_out_3=out.stride()[3],
    )

    return out


def replacement_func():
    return fused_mul_sum_sigmoid