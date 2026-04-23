import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fuse_relu_maxpool_cat_kernel(
    in_ptr,
    out_ptr,
    B, C, H, W,
    stride_in_b, stride_in_c, stride_in_h, stride_in_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_BC: tl.constexpr,
):
    # Grid: (H, W, B*C/BLOCK_BC)
    # Each program handles one spatial output position (h, w) for BLOCK_BC (b, c) pairs
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_bc = tl.program_id(2) * BLOCK_BC

    bc_offsets = pid_bc + tl.arange(0, BLOCK_BC)
    bc_mask = bc_offsets < B * C
    b_idx = bc_offsets // C
    c_idx = bc_offsets % C

    # Clamp b_idx, c_idx for safe memory access even on masked-out elements
    b_idx_safe = tl.where(bc_mask, b_idx, 0)
    c_idx_safe = tl.where(bc_mask, c_idx, 0)

    # Compute 5x5 max_pool2d with stride=1, padding=2, dilation=1
    # Applied AFTER relu (relu of input, then max_pool on relu output)
    # Since relu output >= 0, initialize max_val to 0.0
    max_val = tl.zeros([BLOCK_BC], dtype=tl.float32)

    # 5x5 window: unrolled loop (5 is constexpr)
    for dh in range(5):
        for dw in range(5):
            nh = pid_h + dh - 2
            nw = pid_w + dw - 2

            # Boundary check - valid positions are within [0, H-1] x [0, W-1]
            # Out-of-bounds positions: PyTorch max_pool2d uses -inf padding,
            # but after relu, -inf -> 0, so we use other=0.0
            valid = (nh >= 0) & (nh < H) & (nw >= 0) & (nw < W)

            # Clamp indices for safe memory access
            nh_safe = tl.where(nh < 0, 0, tl.where(nh >= H, H - 1, nh))
            nw_safe = tl.where(nw < 0, 0, tl.where(nw >= W, W - 1, nw))

            offset = b_idx_safe * stride_in_b + c_idx_safe * stride_in_c + nh_safe * stride_in_h + nw_safe * stride_in_w
            # Load input value; for out-of-bounds, use 0.0 (relu(0)=0, won't affect max of >=0 values)
            val = tl.load(in_ptr + offset, mask=bc_mask & valid, other=0.0).to(tl.float32)
            # Apply relu
            relu_val = tl.where(val > 0.0, val, 0.0)
            max_val = tl.maximum(max_val, relu_val)

    # Also load center value for the relu channel (not pooled, just relu)
    center_offset = b_idx_safe * stride_in_b + c_idx_safe * stride_in_c + pid_h * stride_in_h + pid_w * stride_in_w
    center_val = tl.load(in_ptr + center_offset, mask=bc_mask, other=0.0).to(tl.float32)
    relu_center = tl.where(center_val > 0.0, center_val, 0.0)

    # Store 4 output channels: [relu, pool, pool, pool]
    # Since all 3 max_pool2d are identical, store same max_val 3 times
    out_base = b_idx_safe * stride_out_b + pid_h * stride_out_h + pid_w * stride_out_w

    # Cast to output dtype
    out_dtype = out_ptr.dtype.element_ty
    relu_center_out = relu_center.to(out_dtype)
    max_val_out = max_val.to(out_dtype)

    # relu channel (c_idx in [0, C))
    tl.store(out_ptr + out_base + c_idx_safe * stride_out_c, relu_center_out, mask=bc_mask)
    # pool channel 1 (c_idx in [C, 2C))
    tl.store(out_ptr + out_base + (C + c_idx_safe) * stride_out_c, max_val_out, mask=bc_mask)
    # pool channel 2 (c_idx in [2C, 3C))
    tl.store(out_ptr + out_base + (2 * C + c_idx_safe) * stride_out_c, max_val_out, mask=bc_mask)
    # pool channel 3 (c_idx in [3C, 4C))
    tl.store(out_ptr + out_base + (3 * C + c_idx_safe) * stride_out_c, max_val_out, mask=bc_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_BC': 4}, num_warps=2),
        triton.Config({'BLOCK_BC': 8}, num_warps=2),
        triton.Config({'BLOCK_BC': 16}, num_warps=4),
        triton.Config({'BLOCK_BC': 32}, num_warps=4),
        triton.Config({'BLOCK_BC': 64}, num_warps=8),
        triton.Config({'BLOCK_BC': 128}, num_warps=8),
    ],
    key=['B', 'C'],
)
@triton.jit
def fuse_relu_maxpool_cat_kernel_autotuned(
    in_ptr,
    out_ptr,
    B, C, H, W,
    stride_in_b, stride_in_c, stride_in_h, stride_in_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_BC: tl.constexpr,
):
    # Same logic as above, but with autotuned BLOCK_BC
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_bc = tl.program_id(2) * BLOCK_BC

    bc_offsets = pid_bc + tl.arange(0, BLOCK_BC)
    bc_mask = bc_offsets < B * C
    b_idx = bc_offsets // C
    c_idx = bc_offsets % C

    b_idx_safe = tl.where(bc_mask, b_idx, 0)
    c_idx_safe = tl.where(bc_mask, c_idx, 0)

    max_val = tl.zeros([BLOCK_BC], dtype=tl.float32)

    for dh in range(5):
        for dw in range(5):
            nh = pid_h + dh - 2
            nw = pid_w + dw - 2
            valid = (nh >= 0) & (nh < H) & (nw >= 0) & (nw < W)
            nh_safe = tl.where(nh < 0, 0, tl.where(nh >= H, H - 1, nh))
            nw_safe = tl.where(nw < 0, 0, tl.where(nw >= W, W - 1, nw))
            offset = b_idx_safe * stride_in_b + c_idx_safe * stride_in_c + nh_safe * stride_in_h + nw_safe * stride_in_w
            val = tl.load(in_ptr + offset, mask=bc_mask & valid, other=0.0).to(tl.float32)
            relu_val = tl.where(val > 0.0, val, 0.0)
            max_val = tl.maximum(max_val, relu_val)

    center_offset = b_idx_safe * stride_in_b + c_idx_safe * stride_in_c + pid_h * stride_in_h + pid_w * stride_in_w
    center_val = tl.load(in_ptr + center_offset, mask=bc_mask, other=0.0).to(tl.float32)
    relu_center = tl.where(center_val > 0.0, center_val, 0.0)

    out_base = b_idx_safe * stride_out_b + pid_h * stride_out_h + pid_w * stride_out_w
    out_dtype = out_ptr.dtype.element_ty
    relu_center_out = relu_center.to(out_dtype)
    max_val_out = max_val.to(out_dtype)

    tl.store(out_ptr + out_base + c_idx_safe * stride_out_c, relu_center_out, mask=bc_mask)
    tl.store(out_ptr + out_base + (C + c_idx_safe) * stride_out_c, max_val_out, mask=bc_mask)
    tl.store(out_ptr + out_base + (2 * C + c_idx_safe) * stride_out_c, max_val_out, mask=bc_mask)
    tl.store(out_ptr + out_base + (3 * C + c_idx_safe) * stride_out_c, max_val_out, mask=bc_mask)


@torch.fx.wrap
def fuse_relu_maxpool_cat(in_0):
    B, C, H, W = in_0.shape
    out = torch.empty((B, 4 * C, H, W), dtype=in_0.dtype, device=in_0.device)

    grid_h = H
    grid_w = W

    # Use autotuned kernel
    num_bc_programs = triton.cdiv(B * C, 128)  # max BLOCK_BC for grid sizing

    fuse_relu_maxpool_cat_kernel_autotuned[(grid_h, grid_w, num_bc_programs)](
        in_ptr=in_0,
        out_ptr=out,
        B=B, C=C, H=H, W=W,
        stride_in_b=in_0.stride(0), stride_in_c=in_0.stride(1),
        stride_in_h=in_0.stride(2), stride_in_w=in_0.stride(3),
        stride_out_b=out.stride(0), stride_out_c=out.stride(1),
        stride_out_h=out.stride(2), stride_out_w=out.stride(3),
    )

    return out


def replacement_func():
    return fuse_relu_maxpool_cat