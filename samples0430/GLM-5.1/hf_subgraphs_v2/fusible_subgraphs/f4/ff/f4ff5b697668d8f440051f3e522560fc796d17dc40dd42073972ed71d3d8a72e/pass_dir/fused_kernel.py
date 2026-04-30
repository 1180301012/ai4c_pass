import torch
import triton
import triton.language as tl


@triton.jit
def fused_softmax_coord_kernel(
    in_2_ptr, in_0_ptr, in_1_ptr,
    out_softmax_ptr, out_coord_ptr,
    B,
    J: tl.constexpr, HW: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. softmax(in_2, dim=2) viewed as (B, 17, 64, 64)
    2. expected x-coordinate: sum(softmax * x_linspace)
    3. expected y-coordinate: sum(softmax * y_linspace)
    
    Each program handles one (batch, joint) pair.
    """
    bj = tl.program_id(0)
    b = bj // J
    j = bj % J

    # Base pointers for this (b, j) row
    in_2_row = in_2_ptr + b * J * HW + j * HW
    out_row = out_softmax_ptr + b * J * HW + j * HW

    # Pass 1: Find max value for softmax stability
    max_val = -float('inf')
    for start in range(0, HW, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < HW
        x = tl.load(in_2_row + offs, mask=mask, other=0.0)
        max_val = tl.maximum(max_val, tl.max(x, axis=0))

    # Pass 2: Compute exp, accumulate sums, store unnormalized exp
    sum_exp = 0.0
    acc_x = 0.0
    acc_y = 0.0
    for start in range(0, HW, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < HW
        x = tl.load(in_2_row + offs, mask=mask, other=0.0)
        e = tl.exp(x - max_val)

        # Accumulate total exp sum
        sum_exp += tl.sum(e, axis=0)

        # Compute spatial coordinates from flat offset
        h_idx = offs // W
        w_idx = offs % W

        # Load x and y linspace values (both are contiguous 64-element arrays)
        x_lin = tl.load(in_0_ptr + w_idx, mask=mask, other=0.0)
        y_lin = tl.load(in_1_ptr + h_idx, mask=mask, other=0.0)

        # Accumulate weighted sums
        acc_x += tl.sum(e * x_lin, axis=0)
        acc_y += tl.sum(e * y_lin, axis=0)

        # Store unnormalized exp values (will normalize in pass 3)
        tl.store(out_row + offs, e, mask=mask)

    # Compute final coordinates (normalize by sum_exp)
    coord_x = acc_x / sum_exp
    coord_y = acc_y / sum_exp

    # Store coordinates: out_coord[b, j, 0] = x, out_coord[b, j, 1] = y
    coord_base = out_coord_ptr + b * J * 2 + j * 2
    tl.store(coord_base, coord_x)
    tl.store(coord_base + 1, coord_y)

    # Pass 3: Normalize softmax values in-place
    for start in range(0, HW, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < HW
        e = tl.load(out_row + offs, mask=mask, other=0.0)
        tl.store(out_row + offs, e / sum_exp, mask=mask)


@triton.jit
def fused_softmax_coord_kernel_v2(
    in_2_ptr, in_0_ptr, in_1_ptr,
    out_softmax_ptr, out_coord_ptr,
    B,
    J: tl.constexpr, HW: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    """
    Optimized variant with single-pass for small HW.
    Uses a single load when BLOCK_SIZE >= HW.
    """
    bj = tl.program_id(0)
    b = bj // J
    j = bj % J

    in_2_row = in_2_ptr + b * J * HW + j * HW
    out_row = out_softmax_ptr + b * J * HW + j * HW

    # Single pass: load all, compute max, then softmax and weighted sums
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HW
    x = tl.load(in_2_row + offs, mask=mask, other=0.0)

    # Compute max for softmax stability
    max_val = tl.max(x, axis=0)

    # Compute softmax
    e = tl.exp(x - max_val)
    sum_exp = tl.sum(e, axis=0)
    softmax_vals = e / sum_exp

    # Compute spatial coordinates
    h_idx = offs // W
    w_idx = offs % W

    # Load linspace values
    x_lin = tl.load(in_0_ptr + w_idx, mask=mask, other=0.0)
    y_lin = tl.load(in_1_ptr + h_idx, mask=mask, other=0.0)

    # Compute weighted sums
    coord_x = tl.sum(softmax_vals * x_lin, axis=0)
    coord_y = tl.sum(softmax_vals * y_lin, axis=0)

    # Store softmax values
    tl.store(out_row + offs, softmax_vals, mask=mask)

    # Store coordinates
    coord_base = out_coord_ptr + b * J * 2 + j * 2
    tl.store(coord_base, coord_x)
    tl.store(coord_base + 1, coord_y)


@torch.fx.wrap
def fused_softmax_coord(in_0, in_1, in_2, route=""):
    """
    Fused softmax + coordinate computation.
    Replaces the entire subgraph of softmax -> reshape -> mul -> sum -> cat.
    
    route: string identifier for which pattern matched (ignored, same kernel for all).
    """
    B = in_2.shape[0]
    J = in_2.shape[1]  # Should be 17
    H = 64
    W = 64
    HW = H * W  # 4096

    out_softmax = torch.empty(B, J, H, W, dtype=in_2.dtype, device=in_2.device)
    out_coord = torch.empty(B, J, 2, dtype=in_2.dtype, device=in_2.device)

    # Choose kernel variant based on batch size
    # For small batch sizes, use single-pass kernel (BLOCK_SIZE=HW)
    # For larger batch sizes, use multi-pass kernel (BLOCK_SIZE=1024)
    if B * J <= 128:
        # Small workload: use single-pass to minimize loop overhead
        BLOCK_SIZE = HW  # 4096
        grid = (B * J,)
        fused_softmax_coord_kernel_v2[grid](
            in_2, in_0, in_1, out_softmax, out_coord,
            B, J, HW, H, W,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=16,
        )
    else:
        # Larger workload: use multi-pass with smaller block size
        BLOCK_SIZE = 1024
        grid = (B * J,)
        fused_softmax_coord_kernel[grid](
            in_2, in_0, in_1, out_softmax, out_coord,
            B, J, HW, H, W,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out_softmax, out_coord