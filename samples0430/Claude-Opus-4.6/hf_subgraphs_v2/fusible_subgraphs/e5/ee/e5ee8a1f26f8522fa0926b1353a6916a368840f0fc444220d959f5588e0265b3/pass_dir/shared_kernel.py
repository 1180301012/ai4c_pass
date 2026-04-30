import torch
import triton
import triton.language as tl


@triton.jit
def _dw_conv3x3_gelu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    C, H, W, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_hw_blocks = tl.cdiv(HW, BLOCK_SIZE)
    nc_idx = pid // num_hw_blocks
    hw_block = pid % num_hw_blocks

    c = nc_idx % C

    hw_start = hw_block * BLOCK_SIZE
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE)
    mask = hw_offsets < HW

    h_pos = hw_offsets // W
    w_pos = hw_offsets % W

    base_offset = nc_idx * HW

    # Load bias
    b = tl.load(bias_ptr + c).to(tl.float32)
    acc = tl.full([BLOCK_SIZE], b, dtype=tl.float32)

    # Weight base for this channel
    w_base = c * 9

    # 3x3 depthwise conv with padding=1, stride=1, dilation=1
    for kh in tl.static_range(3):
        for kw in tl.static_range(3):
            ih = h_pos + (kh - 1)
            iw = w_pos + (kw - 1)
            valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            inp_idx = base_offset + ih * W + iw
            inp = tl.load(input_ptr + inp_idx, mask=valid, other=0.0).to(tl.float32)
            w_val = tl.load(weight_ptr + w_base + kh * 3 + kw).to(tl.float32)
            acc += inp * w_val

    # Apply GELU
    result = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    # Store output
    out_idx = base_offset + h_pos * W + w_pos
    tl.store(output_ptr + out_idx, result, mask=mask)


@triton.jit
def _dw_conv3x3_gelu_flat_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    C, H, W, HW, total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Flat kernel: processes elements from multiple (n,c) pairs per block.
    Good for small HW where per-channel blocks are too small."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Recover n, c, h, w from flat index (NCHW layout)
    hw = offsets % HW
    nc = offsets // HW
    c = nc % C

    h_pos = hw // W
    w_pos = hw % W

    # Load bias (gathered - each element may have different channel)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    acc = b

    # Weight base per element
    w_base = c * 9
    base_offset = nc * HW

    # 3x3 depthwise conv with padding=1
    for kh in tl.static_range(3):
        for kw in tl.static_range(3):
            ih = h_pos + (kh - 1)
            iw = w_pos + (kw - 1)
            valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            inp_idx = base_offset + ih * W + iw
            inp = tl.load(input_ptr + inp_idx, mask=valid, other=0.0).to(tl.float32)
            w_val = tl.load(weight_ptr + w_base + kh * 3 + kw, mask=mask, other=0.0).to(tl.float32)
            acc += inp * w_val

    # GELU
    result = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _simple_gelu_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    out = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(output_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_dispatch(input, weight, bias, route):
    if route == "dw_conv3x3_gelu":
        N = input.shape[0]
        C = input.shape[1]
        H = input.shape[2]
        W = input.shape[3]
        HW = H * W
        output = torch.empty_like(input)
        if HW <= 256:
            # Use flat kernel for small spatial dims - fewer blocks, better efficiency
            total = N * C * HW
            BLOCK_SIZE = 1024
            grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
            _dw_conv3x3_gelu_flat_kernel[grid](
                input, weight, bias, output, C, H, W, HW, total,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            num_hw_blocks = (HW + 1023) // 1024
            grid = (N * C * num_hw_blocks,)
            _dw_conv3x3_gelu_kernel[grid](
                input, weight, bias, output, C, H, W, HW,
                BLOCK_SIZE=1024,
            )
        return output
    else:
        # simple_gelu route
        output = torch.empty_like(input)
        n_elements = input.numel()
        BLOCK_SIZE = 4096
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _simple_gelu_kernel[grid](input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return output