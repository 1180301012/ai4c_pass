import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_mean_1in_kernel(
    in0_ptr,
    out_sum_ptr,
    out_mean_ptr,
    B, C, H, W,
    stride_in0_b, stride_in0_c, stride_in0_h, stride_in0_w,
    stride_out_sum_b, stride_out_sum_c, stride_out_sum_h, stride_out_sum_w,
    stride_out_mean_b, stride_out_mean_c,
    SPATIAL_BLOCK: tl.constexpr,
):
    """Fused kernel for 1-input: identity + mean over (2,3) with keepdim."""
    bc_idx = tl.program_id(0)
    b = bc_idx // C
    c = bc_idx % C

    base_in0 = b * stride_in0_b + c * stride_in0_c
    base_out_sum = b * stride_out_sum_b + c * stride_out_sum_c

    spatial_size = H * W
    acc_mean = tl.zeros([SPATIAL_BLOCK], dtype=tl.float32)
    total_mean = 0.0

    for spatial_start in tl.range(0, spatial_size, SPATIAL_BLOCK):
        offsets = spatial_start + tl.arange(0, SPATIAL_BLOCK)
        mask = offsets < spatial_size

        # Compute 2D offsets
        h_offsets = offsets // W
        w_offsets = offsets % W

        # Load input
        idx_in0 = base_in0 + h_offsets * stride_in0_h + w_offsets * stride_in0_w
        val = tl.load(in0_ptr + idx_in0, mask=mask, other=0.0).to(tl.float32)

        # Store sum output
        idx_out_sum = base_out_sum + h_offsets * stride_out_sum_h + w_offsets * stride_out_sum_w
        tl.store(out_sum_ptr + idx_out_sum, val.to(out_sum_ptr.dtype.element_ty), mask=mask)

        # Accumulate for mean
        total_mean += tl.sum(val, axis=0)

    # Store mean result at (b, c, 0, 0)
    mean_idx = b * stride_out_mean_b + c * stride_out_mean_c
    mean_val = total_mean / spatial_size
    tl.store(out_mean_ptr + mean_idx, mean_val.to(out_mean_ptr.dtype.element_ty))


@triton.jit
def fused_add_mean_2in_kernel(
    in0_ptr,
    in1_ptr,
    out_sum_ptr,
    out_mean_ptr,
    B, C, H, W,
    stride_in0_b, stride_in0_c, stride_in0_h, stride_in0_w,
    stride_in1_b, stride_in1_c, stride_in1_h, stride_in1_w,
    stride_out_sum_b, stride_out_sum_c, stride_out_sum_h, stride_out_sum_w,
    stride_out_mean_b, stride_out_mean_c,
    SPATIAL_BLOCK: tl.constexpr,
):
    """Fused kernel for 2-input: add + mean over (2,3) with keepdim."""
    bc_idx = tl.program_id(0)
    b = bc_idx // C
    c = bc_idx % C

    base_in0 = b * stride_in0_b + c * stride_in0_c
    base_in1 = b * stride_in1_b + c * stride_in1_c
    base_out_sum = b * stride_out_sum_b + c * stride_out_sum_c

    spatial_size = H * W
    total_mean = 0.0

    for spatial_start in tl.range(0, spatial_size, SPATIAL_BLOCK):
        offsets = spatial_start + tl.arange(0, SPATIAL_BLOCK)
        mask = offsets < spatial_size

        h_offsets = offsets // W
        w_offsets = offsets % W

        # Load inputs
        idx_in0 = base_in0 + h_offsets * stride_in0_h + w_offsets * stride_in0_w
        idx_in1 = base_in1 + h_offsets * stride_in1_h + w_offsets * stride_in1_w
        val0 = tl.load(in0_ptr + idx_in0, mask=mask, other=0.0).to(tl.float32)
        val1 = tl.load(in1_ptr + idx_in1, mask=mask, other=0.0).to(tl.float32)

        # Compute sum
        val_sum = val0 + val1

        # Store sum output
        idx_out_sum = base_out_sum + h_offsets * stride_out_sum_h + w_offsets * stride_out_sum_w
        tl.store(out_sum_ptr + idx_out_sum, val_sum.to(out_sum_ptr.dtype.element_ty), mask=mask)

        # Accumulate for mean
        total_mean += tl.sum(val_sum, axis=0)

    # Store mean result at (b, c, 0, 0)
    mean_idx = b * stride_out_mean_b + c * stride_out_mean_c
    mean_val = total_mean / spatial_size
    tl.store(out_mean_ptr + mean_idx, mean_val.to(out_mean_ptr.dtype.element_ty))


@triton.jit
def fused_add_mean_3in_kernel(
    in0_ptr,
    in1_ptr,
    in2_ptr,
    out_sum_ptr,
    out_mean_ptr,
    B, C, H, W,
    stride_in0_b, stride_in0_c, stride_in0_h, stride_in0_w,
    stride_in1_b, stride_in1_c, stride_in1_h, stride_in1_w,
    stride_in2_b, stride_in2_c, stride_in2_h, stride_in2_w,
    stride_out_sum_b, stride_out_sum_c, stride_out_sum_h, stride_out_sum_w,
    stride_out_mean_b, stride_out_mean_c,
    SPATIAL_BLOCK: tl.constexpr,
):
    """Fused kernel for 3-input: add + add + mean over (2,3) with keepdim."""
    bc_idx = tl.program_id(0)
    b = bc_idx // C
    c = bc_idx % C

    base_in0 = b * stride_in0_b + c * stride_in0_c
    base_in1 = b * stride_in1_b + c * stride_in1_c
    base_in2 = b * stride_in2_b + c * stride_in2_c
    base_out_sum = b * stride_out_sum_b + c * stride_out_sum_c

    spatial_size = H * W
    total_mean = 0.0

    for spatial_start in tl.range(0, spatial_size, SPATIAL_BLOCK):
        offsets = spatial_start + tl.arange(0, SPATIAL_BLOCK)
        mask = offsets < spatial_size

        h_offsets = offsets // W
        w_offsets = offsets % W

        # Load inputs
        idx_in0 = base_in0 + h_offsets * stride_in0_h + w_offsets * stride_in0_w
        idx_in1 = base_in1 + h_offsets * stride_in1_h + w_offsets * stride_in1_w
        idx_in2 = base_in2 + h_offsets * stride_in2_h + w_offsets * stride_in2_w
        val0 = tl.load(in0_ptr + idx_in0, mask=mask, other=0.0).to(tl.float32)
        val1 = tl.load(in1_ptr + idx_in1, mask=mask, other=0.0).to(tl.float32)
        val2 = tl.load(in2_ptr + idx_in2, mask=mask, other=0.0).to(tl.float32)

        # Compute sum
        val_sum = val0 + val1 + val2

        # Store sum output
        idx_out_sum = base_out_sum + h_offsets * stride_out_sum_h + w_offsets * stride_out_sum_w
        tl.store(out_sum_ptr + idx_out_sum, val_sum.to(out_sum_ptr.dtype.element_ty), mask=mask)

        # Accumulate for mean
        total_mean += tl.sum(val_sum, axis=0)

    # Store mean result at (b, c, 0, 0)
    mean_idx = b * stride_out_mean_b + c * stride_out_mean_c
    mean_val = total_mean / spatial_size
    tl.store(out_mean_ptr + mean_idx, mean_val.to(out_mean_ptr.dtype.element_ty))


@torch.fx.wrap
def fused_add_mean_dispatch(*args):
    """Shared dispatch wrapper that routes to the correct kernel based on route string."""
    route = args[-1]
    if route == "1in":
        in0 = args[0]
        B, C, H, W = in0.shape
        out_sum = torch.empty_like(in0)
        out_mean = torch.empty((B, C, 1, 1), dtype=in0.dtype, device=in0.device)

        SPATIAL_BLOCK = 256
        num_programs = B * C

        fused_add_mean_1in_kernel[(num_programs,)](
            in0_ptr=in0,
            out_sum_ptr=out_sum,
            out_mean_ptr=out_mean,
            B=B, C=C, H=H, W=W,
            stride_in0_b=in0.stride(0), stride_in0_c=in0.stride(1),
            stride_in0_h=in0.stride(2), stride_in0_w=in0.stride(3),
            stride_out_sum_b=out_sum.stride(0), stride_out_sum_c=out_sum.stride(1),
            stride_out_sum_h=out_sum.stride(2), stride_out_sum_w=out_sum.stride(3),
            stride_out_mean_b=out_mean.stride(0), stride_out_mean_c=out_mean.stride(1),
            SPATIAL_BLOCK=SPATIAL_BLOCK,
        )
        return (out_sum, out_mean)

    elif route == "2in":
        in0 = args[0]
        in1 = args[1]
        B, C, H, W = in0.shape
        out_sum = torch.empty_like(in0)
        out_mean = torch.empty((B, C, 1, 1), dtype=in0.dtype, device=in0.device)

        SPATIAL_BLOCK = 256
        num_programs = B * C

        fused_add_mean_2in_kernel[(num_programs,)](
            in0_ptr=in0,
            in1_ptr=in1,
            out_sum_ptr=out_sum,
            out_mean_ptr=out_mean,
            B=B, C=C, H=H, W=W,
            stride_in0_b=in0.stride(0), stride_in0_c=in0.stride(1),
            stride_in0_h=in0.stride(2), stride_in0_w=in0.stride(3),
            stride_in1_b=in1.stride(0), stride_in1_c=in1.stride(1),
            stride_in1_h=in1.stride(2), stride_in1_w=in1.stride(3),
            stride_out_sum_b=out_sum.stride(0), stride_out_sum_c=out_sum.stride(1),
            stride_out_sum_h=out_sum.stride(2), stride_out_sum_w=out_sum.stride(3),
            stride_out_mean_b=out_mean.stride(0), stride_out_mean_c=out_mean.stride(1),
            SPATIAL_BLOCK=SPATIAL_BLOCK,
        )
        return (out_sum, out_mean)

    elif route == "3in" or route == "3in_a" or route == "3in_b":
        in0 = args[0]
        in1 = args[1]
        in2 = args[2]
        B, C, H, W = in0.shape
        out_sum = torch.empty_like(in0)
        out_mean = torch.empty((B, C, 1, 1), dtype=in0.dtype, device=in0.device)

        SPATIAL_BLOCK = 256
        num_programs = B * C

        fused_add_mean_3in_kernel[(num_programs,)](
            in0_ptr=in0,
            in1_ptr=in1,
            in2_ptr=in2,
            out_sum_ptr=out_sum,
            out_mean_ptr=out_mean,
            B=B, C=C, H=H, W=W,
            stride_in0_b=in0.stride(0), stride_in0_c=in0.stride(1),
            stride_in0_h=in0.stride(2), stride_in0_w=in0.stride(3),
            stride_in1_b=in1.stride(0), stride_in1_c=in1.stride(1),
            stride_in1_h=in1.stride(2), stride_in1_w=in1.stride(3),
            stride_in2_b=in2.stride(0), stride_in2_c=in2.stride(1),
            stride_in2_h=in2.stride(2), stride_in2_w=in2.stride(3),
            stride_out_sum_b=out_sum.stride(0), stride_out_sum_c=out_sum.stride(1),
            stride_out_sum_h=out_sum.stride(2), stride_out_sum_w=out_sum.stride(3),
            stride_out_mean_b=out_mean.stride(0), stride_out_mean_c=out_mean.stride(1),
            SPATIAL_BLOCK=SPATIAL_BLOCK,
        )
        return (out_sum, out_mean)

    else:
        raise ValueError(f"Unknown route: {route}")