import torch
import triton
import triton.language as tl


@triton.jit
def fused_relu_norm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    norm_scale,
    num_rows,
    row_len,
    x_stride_0,
    x_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= num_rows:
        return

    # Load weight scalar
    weight = tl.load(weight_ptr).to(tl.float32)

    # Compute base offset for this row
    # The original tensor has shape [B, C, H, W] with strides [x_stride_0, x_stride_1, ?, 1]
    # After flatten(2), row_idx maps to batch=row_idx // C_per_batch, channel=row_idx % C_per_batch
    # But since we work with flattened view, we need contiguous access
    # For a contiguous tensor, row_offset = row_idx * row_len
    row_offset = row_idx * row_len
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_len

    # Load input row and convert to float32 for computation
    x = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)

    # Apply ReLU
    x_relu = tl.maximum(x, 0.0)

    # Compute squared sum for L2 norm along last dimension
    sq_sum = tl.sum(x_relu * x_relu)

    # Compute norm: sqrt(sq_sum) * norm_scale, then clamp
    norm_val = tl.sqrt(sq_sum) * norm_scale
    norm_val = tl.maximum(norm_val, 1e-05)

    # Normalize: divide by clamped norm, then multiply by weight
    out = (x_relu / norm_val) * weight

    # Store result
    tl.store(out_ptr + row_offset + offsets, out, mask=mask)


@torch.fx.wrap
def fused_relu_norm_dispatch(in_0, in_1, route):
    """Dispatch wrapper that routes to the appropriate norm_scale based on route string."""
    # Determine norm_scale based on route
    if route == "route_1443":
        norm_scale_val = 0.14433756729740643
    elif route == "route_0721":
        norm_scale_val = 0.07216878364870322
    else:
        raise ValueError(f"Unknown route: {route}")

    # Compute flattened dimensions
    # flatten(start_dim=2) preserves dims 0,1 and flattens dims 2+
    num_rows = in_1.shape[0] * in_1.shape[1]
    row_len = 1
    for d in in_1.shape[2:]:
        row_len *= d

    # Create output tensor with flattened shape
    out = torch.empty(num_rows, row_len, dtype=in_1.dtype, device=in_1.device)

    # Ensure input is contiguous for correct memory access
    if not in_1.is_contiguous():
        in_1 = in_1.contiguous()

    # Compute BLOCK_SIZE (must be power of 2, >= row_len)
    BLOCK_SIZE = max(64, triton.next_power_of_2(row_len))

    # Launch kernel: one program per row
    grid = (num_rows,)
    fused_relu_norm_kernel[grid](
        x_ptr=in_1,
        weight_ptr=in_0,
        out_ptr=out,
        norm_scale=norm_scale_val,
        num_rows=num_rows,
        row_len=row_len,
        x_stride_0=in_1.stride()[0],
        x_stride_1=in_1.stride()[1],
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out