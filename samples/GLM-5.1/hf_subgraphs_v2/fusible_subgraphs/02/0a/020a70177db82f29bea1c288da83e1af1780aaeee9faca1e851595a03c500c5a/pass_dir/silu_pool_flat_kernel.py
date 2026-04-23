import torch
import triton
import triton.language as tl

@triton.jit
def silu_pool_flat_kernel(
    input_ptr,
    output_ptr,
    N, C, H, W,
    HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """
    Fused SiLU + AdaptiveAvgPool2d(1) + Flatten(1) kernel.
    
    Input shape: [N, C, H, W]
    Output shape: [N, C] (after adaptive_avg_pool2d → [N, C, 1, 1] and flatten → [N, C])
    
    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    AdaptiveAvgPool2d(1) reduces spatial dims by averaging: mean over H*W
    """
    # Each program handles one (n_batch, block_of_channels) pair
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    c_start = pid_c * BLOCK_C
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C
    
    # Output offset: output is [N, C]
    out_offset = pid_n * C + c_offsets
    
    # Accumulate silu values over spatial dimensions
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    
    for h_w_start in range(0, HW, BLOCK_HW):
        hw_offsets = h_w_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW
        
        # Compute h and w from flat HW index
        h_offsets = hw_offsets // W
        w_offsets = hw_offsets % W
        
        # Input offset: [N, C, H, W] - strided access
        # input_ptr + n * (C*H*W) + c * (H*W) + h * W + w
        in_offset = pid_n * (C * HW) + c_offsets * HW + hw_offsets
        
        # Load with 2D mask: c_mask AND hw_mask broadcast properly
        mask = c_mask & hw_mask  # broadcast: [BLOCK_C] & [BLOCK_HW] → needs reshaping
        
        # Actually we need to handle this differently - load per hw element across channels
        # We need to load BLOCK_C values for each of BLOCK_HW spatial positions
        # But tl.load with 2D indexing is complex. Let's reshape.
        
        # 2D offsets: [BLOCK_C, BLOCK_HW]
        offsets_2d = c_offsets[:, None] * HW + hw_offsets[None, :]  # [BLOCK_C, BLOCK_HW]
        base = pid_n * C * HW
        mask_2d = c_mask[:, None] & hw_mask[None, :]  # [BLOCK_C, BLOCK_HW]
        
        x = tl.load(input_ptr + base + offsets_2d, mask=mask_2d, other=0.0)  # [BLOCK_C, BLOCK_HW]
        
        # Apply SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        silu_val = x * tl.sigmoid(x)  # [BLOCK_C, BLOCK_HW]
        
        # Accumulate across spatial dimension
        acc += tl.sum(silu_val, axis=1)  # sum over BLOCK_HW axis → [BLOCK_C]
    
    # Divide by HW (number of spatial elements)
    result = acc / HW
    
    # Store to output [N, C]
    # Cast back to original dtype
    out_dtype = output_ptr.dtype.element_ty
    tl.store(output_ptr + out_offset, result.to(out_dtype), mask=c_mask)


@triton.jit
def silu_pool_flat_kernel_v2(
    input_ptr,
    output_ptr,
    N, C, H, W,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused SiLU + AdaptiveAvgPool2d(1) + Flatten(1) kernel - version 2.
    Process output elements in blocks of BLOCK_SIZE.
    Each block handles a contiguous range of output elements (N*C total).
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    total = N * C
    mask = offsets < total
    
    # Decode offset into (n, c) indices
    n_idx = offsets // C
    c_idx = offsets % C
    
    # Accumulate silu values over all spatial positions
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Iterate over spatial positions
    for hw in range(HW):
        h_idx = hw // W
        w_idx = hw % W
        
        # Load input at [n, c, h, w]
        in_offsets = n_idx * (C * H * W) + c_idx * (H * W) + h_idx * W + w_idx
        
        x = tl.load(input_ptr + in_offsets, mask=mask, other=0.0)
        
        # SiLU
        silu_val = x * tl.sigmoid(x)
        
        acc += silu_val
    
    result = acc / HW
    
    out_dtype = output_ptr.dtype.element_ty
    tl.store(output_ptr + offsets, result.to(out_dtype), mask=mask)


@triton.jit
def silu_pool_flat_kernel_v3(
    input_ptr,
    output_ptr,
    N, C, H, W,
    HW,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """
    Fused SiLU + AdaptiveAvgPool2d(1) + Flatten(1) kernel - version 3.
    Two-level tiling for better memory access patterns.
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    c_start = pid_c * BLOCK_C
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C
    
    # Output offset: output is [N, C]
    out_offset = pid_n * C + c_offsets
    
    # Accumulate silu values over spatial dimensions
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    
    for h_w_start in range(0, HW, BLOCK_HW):
        hw_offsets = h_w_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW
        
        # 2D offsets: [BLOCK_C, BLOCK_HW]
        offsets_2d = c_offsets[:, None] * HW + hw_offsets[None, :]
        base = pid_n * C * HW
        mask_2d = c_mask[:, None] & hw_mask[None, :]
        
        x = tl.load(input_ptr + base + offsets_2d, mask=mask_2d, other=0.0)
        
        # SiLU
        silu_val = x * tl.sigmoid(x)
        
        # Accumulate across spatial dimension
        acc += tl.sum(silu_val, axis=1)
    
    result = acc / HW
    
    out_dtype = output_ptr.dtype.element_ty
    tl.store(output_ptr + out_offset, result.to(out_dtype), mask=c_mask)


def launch_silu_pool_flat(input_tensor, route_str):
    """
    Launch the fused SiLU + adaptive_avg_pool2d(1) + flatten(1) kernel.
    Since dropout with training=False is identity, we don't need to apply it.
    """
    N, C, H, W = input_tensor.shape
    HW = H * W
    
    output = torch.empty((N, C), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Choose kernel version based on problem size
    total_output = N * C
    
    if HW <= 64 and total_output >= 256:
        # Use v2 kernel with large block size
        BLOCK_SIZE = 256
        grid = ((total_output + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        silu_pool_flat_kernel_v2[grid](
            input_ptr=input_tensor,
            output_ptr=output,
            N=N, C=C, H=H, W=W,
            HW=HW,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Use v3 kernel with 2D grid
        BLOCK_C = min(32, C)
        BLOCK_HW = min(64, HW)
        grid_c = (C + BLOCK_C - 1) // BLOCK_C
        grid = (N, grid_c)
        silu_pool_flat_kernel_v3[grid](
            input_ptr=input_tensor,
            output_ptr=output,
            N=N, C=C, H=H, W=W,
            HW=HW,
            BLOCK_C=BLOCK_C,
            BLOCK_HW=BLOCK_HW,
        )
    
    return output


@torch.fx.wrap
def fused_silu_pool_flat_dispatch(input_tensor, route_str):
    """Dispatch wrapper for the shared replacement_func routing technique."""
    return launch_silu_pool_flat(input_tensor, route_str)