import torch
import triton
import triton.language as tl


@triton.jit
def fused_key_kernel(
    input_ptr, weight_ptr, key_out_ptr,
    C_in: tl.int32, H: tl.int32, W: tl.int32,
    head_dim: tl.int32,
    num_heads: tl.constexpr, key_dim: tl.constexpr,
    n_windows: tl.constexpr, n_windows_h: tl.constexpr, n_windows_w: tl.constexpr,
    kernel_size: tl.constexpr, stride_size: tl.constexpr, pad_size: tl.constexpr,
    n_spatial: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_head = tl.program_id(0)
    pid_win = tl.program_id(1)
    pid_kh = tl.program_id(2)

    # Compute window indices
    win_i = pid_win // n_windows_w
    win_j = pid_win % n_windows_w

    # Compute original spatial positions
    h_orig = win_i * stride_size + pid_kh - pad_size
    w_orig_base = win_j * stride_size - pad_size

    # Check if h_orig is in bounds
    h_in_bounds = (h_orig >= 0) & (h_orig < H)

    # Output channel offset for this head (key channels only)
    ch_offset = pid_head * head_dim

    # Accumulator for key output: [key_dim, kernel_size]
    acc = tl.zeros([key_dim, kernel_size], dtype=tl.float32)

    # Reduction loop over C_in
    num_k_blocks = tl.cdiv(C_in, BLOCK_K)
    for k_block_idx in tl.range(0, num_k_blocks):
        k_start = k_block_idx * BLOCK_K
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < C_in

        # Load weight block: [key_dim, BLOCK_K]
        # weight[ch_offset + key_ch, k] for key_ch in 0..key_dim-1
        key_ch = tl.arange(0, key_dim)
        weight_offsets = (ch_offset + key_ch[:, None]) * C_in + k_offsets[None, :]
        weight_mask = k_mask[None, :]
        weight_block = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)

        # Load input block: [BLOCK_K, kernel_size]
        # input[k, h_orig, w_orig_base + kw] for kw in 0..kernel_size-1
        kw = tl.arange(0, kernel_size)
        w_orig = w_orig_base + kw
        w_in_bounds = (w_orig >= 0) & (w_orig < W)

        # input layout: [1, C_in, H, W], offset = k * H * W + h_orig * W + w_orig
        input_offsets = k_offsets[:, None] * (H * W) + h_orig * W + w_orig[None, :]
        input_mask = k_mask[:, None] & h_in_bounds & w_in_bounds[None, :]
        input_block = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)

        # Matmul: weight_block [key_dim, BLOCK_K] @ input_block [BLOCK_K, kernel_size]
        acc += tl.dot(weight_block, input_block)

    # Store key output: key_out[num_heads, n_windows, key_dim, n_spatial]
    spatial_base = pid_kh * kernel_size

    key_ch_store = tl.arange(0, key_dim)
    kw_store = tl.arange(0, kernel_size)

    # offset = head * (n_windows * key_dim * n_spatial) + win * (key_dim * n_spatial) + key_ch * n_spatial + spatial
    key_out_base = pid_head * (n_windows * key_dim * n_spatial) + pid_win * (key_dim * n_spatial)
    key_out_offsets = key_out_base + key_ch_store[:, None] * n_spatial + (spatial_base + kw_store[None, :])

    tl.store(key_out_ptr + key_out_offsets, acc)


@triton.jit
def fused_val_kernel(
    input_ptr, weight_ptr, val_out_ptr,
    C_in: tl.int32, H: tl.int32, W: tl.int32,
    head_dim: tl.int32, key_dim: tl.constexpr,
    num_heads: tl.constexpr, val_dim: tl.constexpr,
    n_windows: tl.constexpr, n_windows_h: tl.constexpr, n_windows_w: tl.constexpr,
    kernel_size: tl.constexpr, stride_size: tl.constexpr, pad_size: tl.constexpr,
    n_spatial: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_head = tl.program_id(0)
    pid_win = tl.program_id(1)
    pid_kh = tl.program_id(2)

    # Compute window indices
    win_i = pid_win // n_windows_w
    win_j = pid_win % n_windows_w

    # Compute original spatial positions
    h_orig = win_i * stride_size + pid_kh - pad_size
    w_orig_base = win_j * stride_size - pad_size

    # Check if h_orig is in bounds
    h_in_bounds = (h_orig >= 0) & (h_orig < H)

    # Output channel offset for this head (value channels: after key channels)
    ch_offset = pid_head * head_dim + key_dim

    # Accumulator for value output: [val_dim, kernel_size]
    acc = tl.zeros([val_dim, kernel_size], dtype=tl.float32)

    # Reduction loop over C_in
    num_k_blocks = tl.cdiv(C_in, BLOCK_K)
    for k_block_idx in tl.range(0, num_k_blocks):
        k_start = k_block_idx * BLOCK_K
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < C_in

        # Load weight block: [val_dim, BLOCK_K]
        # weight[ch_offset + val_ch, k] for val_ch in 0..val_dim-1
        val_ch = tl.arange(0, val_dim)
        weight_offsets = (ch_offset + val_ch[:, None]) * C_in + k_offsets[None, :]
        weight_mask = k_mask[None, :]
        weight_block = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)

        # Load input block: [BLOCK_K, kernel_size]
        # Same as key kernel
        kw = tl.arange(0, kernel_size)
        w_orig = w_orig_base + kw
        w_in_bounds = (w_orig >= 0) & (w_orig < W)

        input_offsets = k_offsets[:, None] * (H * W) + h_orig * W + w_orig[None, :]
        input_mask = k_mask[:, None] & h_in_bounds & w_in_bounds[None, :]
        input_block = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)

        # Matmul: weight_block [val_dim, BLOCK_K] @ input_block [BLOCK_K, kernel_size]
        acc += tl.dot(weight_block, input_block)

    # Store value output: val_out[num_heads, n_windows, n_spatial, val_dim]
    spatial_base = pid_kh * kernel_size

    val_ch_store = tl.arange(0, val_dim)
    kw_store = tl.arange(0, kernel_size)

    # val_out layout: [num_heads, n_windows, n_spatial, val_dim]
    # offset = head * (n_windows * n_spatial * val_dim) + win * (n_spatial * val_dim) + spatial * val_dim + val_ch
    # For [val_dim, kernel_size] accumulator matching [val_ch, kw]:
    # offsets[val_ch, kw] = base + (spatial_base + kw) * val_dim + val_ch
    val_out_base = pid_head * (n_windows * n_spatial * val_dim) + pid_win * (n_spatial * val_dim)
    val_out_offsets = val_out_base + (spatial_base + kw_store[None, :]) * val_dim + val_ch_store[:, None]

    tl.store(val_out_ptr + val_out_offsets, acc)


def _launch_kernels(weight, input, num_heads, head_dim, key_dim, val_dim, H, W, C_in,
                     n_windows, n_windows_h, n_windows_w, kernel_size, stride_size, pad_size,
                     BLOCK_K=64):
    """Common kernel launch logic shared by both route variants."""
    n_spatial = kernel_size * kernel_size  # 144

    # Ensure tensors are on the same device and contiguous
    weight = weight.to(input.device).contiguous()
    input = input.contiguous()

    # Allocate output tensors
    key_out = torch.empty((num_heads, n_windows, key_dim, n_spatial), dtype=input.dtype, device=input.device)
    val_out = torch.empty((num_heads, n_windows, n_spatial, val_dim), dtype=input.dtype, device=input.device)

    # Grid dimensions: (num_heads, n_windows, kernel_size)
    grid = (num_heads, n_windows, kernel_size)

    # Launch key kernel
    fused_key_kernel[grid](
        input_ptr=input, weight_ptr=weight, key_out_ptr=key_out,
        C_in=C_in, H=H, W=W,
        head_dim=head_dim,
        num_heads=num_heads, key_dim=key_dim,
        n_windows=n_windows, n_windows_h=n_windows_h, n_windows_w=n_windows_w,
        kernel_size=kernel_size, stride_size=stride_size, pad_size=pad_size,
        n_spatial=n_spatial,
        BLOCK_K=BLOCK_K,
    )

    # Launch value kernel
    fused_val_kernel[grid](
        input_ptr=input, weight_ptr=weight, val_out_ptr=val_out,
        C_in=C_in, H=H, W=W,
        head_dim=head_dim, key_dim=key_dim,
        num_heads=num_heads, val_dim=val_dim,
        n_windows=n_windows, n_windows_h=n_windows_h, n_windows_w=n_windows_w,
        kernel_size=kernel_size, stride_size=stride_size, pad_size=pad_size,
        n_spatial=n_spatial,
        BLOCK_K=BLOCK_K,
    )

    return (key_out, val_out)


@torch.fx.wrap
def fused_kernel_16_64(weight, input):
    """Float16 variant: C_in=512, head_dim=80, key_dim=16, val_dim=64."""
    C_in = weight.shape[1]
    H, W = input.shape[2], input.shape[3]
    return _launch_kernels(
        weight, input,
        num_heads=8, head_dim=80, key_dim=16, val_dim=64,
        H=H, W=W, C_in=C_in,
        n_windows=4, n_windows_h=2, n_windows_w=2,
        kernel_size=12, stride_size=8, pad_size=2,
        BLOCK_K=64,
    )


@torch.fx.wrap
def fused_kernel_16_32(weight, input):
    """Bfloat16/float32 variant: C_in=256, head_dim=48, key_dim=16, val_dim=32."""
    C_in = weight.shape[1]
    H, W = input.shape[2], input.shape[3]
    return _launch_kernels(
        weight, input,
        num_heads=8, head_dim=48, key_dim=16, val_dim=32,
        H=H, W=W, C_in=C_in,
        n_windows=4, n_windows_h=2, n_windows_w=2,
        kernel_size=12, stride_size=8, pad_size=2,
        BLOCK_K=64,
    )


@torch.fx.wrap
def dispatch_wrapper(weight, input, route):
    """Shared dispatch wrapper for both route variants."""
    if route == "route_16_64":
        return fused_kernel_16_64(weight, input)
    elif route == "route_16_32":
        return fused_kernel_16_32(weight, input)
    else:
        raise ValueError(f"Unknown route: {route}")