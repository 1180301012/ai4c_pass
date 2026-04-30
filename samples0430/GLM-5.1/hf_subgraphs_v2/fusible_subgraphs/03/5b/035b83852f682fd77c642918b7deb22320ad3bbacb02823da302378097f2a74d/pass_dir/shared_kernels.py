import torch
import triton
import triton.language as tl


# ===== Triton Layer Norm Kernel =====

@triton.jit
def layer_norm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    stride_x, stride_y,
    D, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_x
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    mask_float = tl.where(mask, 1.0, 0.0)

    # Load row as float32 for numerical stability
    x = tl.load(X_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute mean (only over valid elements)
    mean = tl.sum(x * mask_float, axis=0) / D

    # Compute variance (only over valid elements)
    diff = (x - mean) * mask_float
    var = tl.sum(diff * diff, axis=0) / D

    # Normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * rstd

    # Apply weight and bias
    w = tl.load(W_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    y = x_norm * w + b

    # Store
    tl.store(Y_ptr + row_start + offsets, y, mask=mask)


# ===== Triton Relative Position Bias Kernel =====

@triton.jit
def rel_pos_bias_kernel(
    output_ptr,
    GRID_SIZE: tl.constexpr,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL

    # Decode flat index to (a, b, c)
    c = offsets % 3
    rem = offsets // 3
    b_idx = rem % N
    a_idx = rem // N

    # Compute grid coordinates
    col_a = a_idx % GRID_SIZE
    row_a = a_idx // GRID_SIZE
    col_b = b_idx % GRID_SIZE
    row_b = b_idx // GRID_SIZE

    # Compute relative distances as float32
    dx = (col_b - col_a).to(tl.float32)
    dy = (row_b - row_a).to(tl.float32)
    d2 = dx * dx + dy * dy

    # Select value based on channel
    result = tl.where(c == 0, dx, tl.where(c == 1, dy, d2))

    tl.store(output_ptr + offsets, result, mask=mask)


# ===== Implementation Functions =====

def _layer_norm_impl(x, weight, bias, D, eps):
    """Layer norm implementation."""
    N = x.shape[1]
    y = torch.empty_like(x)

    BLOCK_SIZE = 512

    grid = (N,)
    layer_norm_kernel[grid](
        X_ptr=x, W_ptr=weight, B_ptr=bias, Y_ptr=y,
        stride_x=x.stride(1), stride_y=y.stride(1),
        D=D, eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


def _rel_pos_bias_impl(device):
    """Position bias implementation."""
    output = torch.zeros(1, 196, 196, 3, device=device, dtype=torch.float32)
    TOTAL = 196 * 196 * 3
    BLOCK_SIZE = 1024
    num_programs = (TOTAL + BLOCK_SIZE - 1) // BLOCK_SIZE

    rel_pos_bias_kernel[(num_programs,)](
        output_ptr=output,
        GRID_SIZE=14, N=196, TOTAL=TOTAL,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


# ===== Shared Dispatch Wrapper =====
# Pattern only matches layer_norm, so replacement only returns layer_norm result

@torch.fx.wrap
def fused_convit_dispatch(in_0, in_1, in_2, route):
    if route == "route_192":
        return _layer_norm_impl(in_2, in_1, in_0, 192, 1e-06)
    elif route == "route_432":
        return _layer_norm_impl(in_2, in_1, in_0, 432, 1e-06)
    else:
        raise ValueError(f"Unknown route: {route}")