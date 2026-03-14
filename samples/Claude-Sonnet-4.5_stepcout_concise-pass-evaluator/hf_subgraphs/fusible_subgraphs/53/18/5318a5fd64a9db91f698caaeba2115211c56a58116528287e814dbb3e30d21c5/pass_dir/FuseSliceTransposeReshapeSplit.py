import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """Pattern matching the entire forward pass for coat_lite_medium_384_start26_end35_5"""
    tmp_0 = in_1 @ in_0
    tmp_1 = in_1[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_2 = in_2[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, 128, 96, 96)
    tmp_5 = torch.functional.split(tmp_4, [32, 48, 48], dim=1)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    return (tmp_0, tmp_6, tmp_7, tmp_8, tmp_1)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_slice_transpose_reshape_split_kernel(
    input_ptr,
    out1_ptr,
    out2_ptr,
    out3_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    HH: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: slice[:,:,1:,:] -> transpose(-1,-2) -> reshape -> split
    Input: [B, H, N, D], we slice to get [B, H, N-1, D]
    After transpose: [B, H, D, N-1]
    After reshape: [B, H*D, HH, HH] where HH*HH = N-1
    Split into 3 tensors of size [B, C1, HH, HH], [B, C2, HH, HH], [B, C3, HH, HH]
    """
    pid = tl.program_id(0)
    total_elements = B * (C1 + C2 + C3) * HH * HH
    
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    
    # Decode output position: [b, c, h, w]
    w = idx % HH
    h = (idx // HH) % HH
    c = (idx // (HH * HH)) % (C1 + C2 + C3)
    b = idx // ((C1 + C2 + C3) * HH * HH)
    
    # Map back to input indices
    # After reshape, the mapping is:
    # output[b, c, h, w] corresponds to reshaped[b, c, h, w]
    # where reshaped came from transposed[b, head, dim, seq]
    # and c = head * D + dim, h*HH + w is the linear index in sequence
    
    head = c // D
    dim = c % D
    seq_idx = h * HH + w
    
    # In the original input (after slice), this maps to:
    # input[b, head, seq_idx + 1, dim] (the +1 is from slicing at index 1)
    input_idx = (b * H * N * D + 
                 head * N * D + 
                 (seq_idx + 1) * D + 
                 dim)
    
    valid = (b < B) & (head < H) & (seq_idx < (N - 1)) & (dim < D)
    mask = mask & valid
    
    # Load from input (with transpose and slice)
    value = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    
    # Store to appropriate output based on channel index
    output_idx = b * HH * HH + h * HH + w
    
    # Determine which output tensor
    if c < C1:
        # First split
        out_idx = b * C1 * HH * HH + c * HH * HH + h * HH + w
        tl.store(out1_ptr + out_idx, value, mask=mask & (c < C1))
    elif c < (C1 + C2):
        # Second split
        c_offset = c - C1
        out_idx = b * C2 * HH * HH + c_offset * HH * HH + h * HH + w
        tl.store(out2_ptr + out_idx, value, mask=mask)
    else:
        # Third split
        c_offset = c - C1 - C2
        out_idx = b * C3 * HH * HH + c_offset * HH * HH + h * HH + w
        tl.store(out3_ptr + out_idx, value, mask=mask)


@torch.fx.wrap
def fused_forward(in_0, in_1, in_2):
    """
    Optimized forward pass with fused slice-transpose-reshape-split for in_2
    """
    # Matmul (let cuBLAS handle this)
    tmp_0 = in_1 @ in_0
    
    # Slice from in_1
    tmp_1 = in_1[:, :, 1:, :]
    
    # Fused operation for in_2
    B, H, N, D = in_2.shape
    
    # After slice: [B, H, N-1, D]
    # After transpose: [B, H, D, N-1]
    # After reshape: [B, H*D, HH, HH] where HH^2 = N-1
    HH = int((N - 1) ** 0.5)
    C_total = H * D
    
    # Determine split sizes based on pattern
    # Pattern: [C/4, 3C/8, 3C/8]
    C1 = C_total // 4
    C2 = (C_total - C1) // 2
    C3 = C_total - C1 - C2
    
    # Allocate outputs
    tmp_6 = torch.empty((B, C1, HH, HH), device=in_2.device, dtype=in_2.dtype)
    tmp_7 = torch.empty((B, C2, HH, HH), device=in_2.device, dtype=in_2.dtype)
    tmp_8 = torch.empty((B, C3, HH, HH), device=in_2.device, dtype=in_2.dtype)
    
    # Launch kernel
    total_elements = B * C_total * HH * HH
    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_slice_transpose_reshape_split_kernel[grid](
        in_2,
        tmp_6,
        tmp_7,
        tmp_8,
        B=B,
        H=H,
        N=N,
        D=D,
        HH=HH,
        C1=C1,
        C2=C2,
        C3=C3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (tmp_0, tmp_6, tmp_7, tmp_8, tmp_1)


def replacement_func():
    return fused_forward