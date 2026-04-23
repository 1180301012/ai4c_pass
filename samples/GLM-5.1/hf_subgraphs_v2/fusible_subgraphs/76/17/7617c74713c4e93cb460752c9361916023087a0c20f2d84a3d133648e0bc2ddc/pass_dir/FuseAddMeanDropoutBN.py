import torch
import triton
import triton.language as tl


def pattern(in_4: torch.Tensor, in_5):
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    return tmp_5


def replacement_args(in_4, in_5):
    return (in_4, in_5)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 64}, num_warps=2),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 128}, num_warps=2),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 128}, num_warps=8),
        triton.Config({'BLOCK_C': 128, 'BLOCK_HW': 64}, num_warps=8),
        triton.Config({'BLOCK_C': 128, 'BLOCK_HW': 128}, num_warps=8),
        triton.Config({'BLOCK_C': 256, 'BLOCK_HW': 64}, num_warps=8),
        triton.Config({'BLOCK_C': 256, 'BLOCK_HW': 128}, num_warps=16),
    ],
    key=['channels', 'spatial_h', 'spatial_w'],
)
@triton.jit
def fused_add_mean_kernel(
    in_4_ptr, in_5_ptr,
    mean_out_ptr,
    batch_size, channels, spatial_h, spatial_w,
    in_4_stride_b, in_4_stride_c, in_4_stride_h, in_4_stride_w,
    in_5_stride_b, in_5_stride_c, in_5_stride_h, in_5_stride_w,
    mean_stride_b, mean_stride_c,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_offset = pid_c * BLOCK_C
    c_offsets = c_offset + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < channels

    spatial_size = spatial_h * spatial_w
    n_hw_blocks = (spatial_size + BLOCK_HW - 1) // BLOCK_HW

    mean_vals = tl.zeros([BLOCK_C], dtype=tl.float32)

    for hw_block in range(n_hw_blocks):
        hw_offset = hw_block * BLOCK_HW
        hw_offsets = hw_offset + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < spatial_size

        h_idx = hw_offsets // spatial_w
        w_idx = hw_offsets % spatial_w

        # Load in_4 and in_5
        in_4_ptrs = in_4_ptr + pid_b * in_4_stride_b + c_offsets[:, None] * in_4_stride_c + h_idx[None, :] * in_4_stride_h + w_idx[None, :] * in_4_stride_w
        in_5_ptrs = in_5_ptr + pid_b * in_5_stride_b + c_offsets[:, None] * in_5_stride_c + h_idx[None, :] * in_5_stride_h + w_idx[None, :] * in_5_stride_w

        in_4_vals = tl.load(in_4_ptrs, mask=c_mask[:, None] & hw_mask[None, :], other=0.0)
        in_5_vals = tl.load(in_5_ptrs, mask=c_mask[:, None] & hw_mask[None, :], other=0.0)

        add_vals = in_4_vals + in_5_vals
        mean_vals += tl.sum(add_vals, axis=1)

    mean_vals = mean_vals / spatial_size

    mean_ptrs = mean_out_ptr + pid_b * mean_stride_b + c_offsets * mean_stride_c
    tl.store(mean_ptrs, mean_vals, mask=c_mask)


@torch.fx.wrap
def fused_add_mean(in_4, in_5):
    B, C, H, W = in_4.shape

    mean_out = torch.empty(B, C, dtype=in_4.dtype, device=in_4.device)

    # Use BLOCK_C that divides C or covers all channels in one program
    BLOCK_C = min(C, 64)  # autotune will override this
    grid = (B, triton.cdiv(C, BLOCK_C))

    fused_add_mean_kernel[grid](
        in_4_ptr=in_4, in_5_ptr=in_5,
        mean_out_ptr=mean_out,
        batch_size=B, channels=C, spatial_h=H, spatial_w=W,
        in_4_stride_b=in_4.stride()[0], in_4_stride_c=in_4.stride()[1],
        in_4_stride_h=in_4.stride()[2], in_4_stride_w=in_4.stride()[3],
        in_5_stride_b=in_5.stride()[0], in_5_stride_c=in_5.stride()[1],
        in_5_stride_h=in_5.stride()[2], in_5_stride_w=in_5.stride()[3],
        mean_stride_b=mean_out.stride()[0], mean_stride_c=mean_out.stride()[1],
    )

    return mean_out


def replacement_func():
    return fused_add_mean