import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_3, in_4):
    """
    Pattern: add → layer_norm → reshape → permute → contiguous (variant 2)
    Uses in_4 + in_3 for add
    """
    tmp_2 = in_4 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (512,), in_1, in_0, 1e-05)
    tmp_4 = tmp_3.reshape(1, 16, 16, -1)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_3, in_4):
    return (in_0, in_1, in_3, in_4)


@triton.jit
def fused_add_layernorm_reshape_permute_kernel(
    in_3_ptr, in_4_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N,
    eps,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
):
    """
    Fused kernel for add + layer_norm + reshape + permute + contiguous
    Input: [B*256, 512]
    Output: [B, 512, 16, 16]
    Process one row at a time
    """
    BLOCK_SIZE: tl.constexpr = 1024  # Must be >= 512
    
    row_idx = tl.program_id(0)
    
    if row_idx >= M:
        return
    
    # Load the row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    in_3_row = tl.load(in_3_ptr + row_idx * N + offs, mask=mask, other=0.0)
    in_4_row = tl.load(in_4_ptr + row_idx * N + offs, mask=mask, other=0.0)
    
    # Add
    x = in_4_row + in_3_row
    
    # Layer norm
    x_valid = tl.where(mask, x, 0.0)
    mean = tl.sum(x_valid) / 512.0
    
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered) / 512.0
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + offs, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    
    # Normalize and scale
    normalized = x_centered * rstd
    output = tl.where(mask, normalized * weight + bias, 0.0)
    
    # Reshape and permute: [B*256, 512] → [B, 16, 16, 512] → [B, 512, 16, 16]
    batch = row_idx // 256
    spatial_pos = row_idx % 256
    h = spatial_pos // 16
    w = spatial_pos % 16
    
    # Output position: [batch, channel, h, w]
    out_base = batch * stride_out_b + h * stride_out_h + w * stride_out_w
    out_ptrs = out_ptr + out_base + offs * stride_out_c
    tl.store(out_ptrs, output, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256}, num_warps=8),
    ],
    key=['B', 'C', 'HW'],
)
@triton.jit
def fused_flatten_transpose_kernel(
    in_ptr, out_ptr,
    B, C, HW,
    stride_in_b, stride_in_c, stride_in_hw,
    stride_out_b, stride_out_hw, stride_out_c,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel for flatten + transpose
    Input: [B, C, H, W] → flatten(2) → [B, C, H*W] → transpose(1, 2) → [B, H*W, C]
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Calculate indices
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_hw = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Mask for valid indices
    mask_b = offs_b < B
    mask_hw = offs_hw < HW
    mask_c = offs_c < C
    
    # Create 3D mask
    mask = mask_b[:, None, None] & mask_hw[None, :, None] & mask_c[None, None, :]
    
    # Input: [B, C, HW] (already flattened conceptually)
    # Calculate input pointers
    in_ptrs = (in_ptr + 
               offs_b[:, None, None] * stride_in_b + 
               offs_c[None, None, :] * stride_in_c + 
               offs_hw[None, :, None] * stride_in_hw)
    
    # Load data
    data = tl.load(in_ptrs, mask=mask, other=0.0)
    
    # Output: [B, HW, C] (transposed)
    # Calculate output pointers
    out_ptrs = (out_ptr + 
                offs_b[:, None, None] * stride_out_b + 
                offs_hw[None, :, None] * stride_out_hw + 
                offs_c[None, None, :] * stride_out_c)
    
    # Store data
    tl.store(out_ptrs, data, mask=mask)


@torch.fx.wrap
def fused_add_layernorm_reshape_permute_variant2(in_0, in_1, in_3, in_4):
    """
    Fused implementation of add + layer_norm + reshape + permute + contiguous (variant 2)
    in_0: bias [512]
    in_1: weight [512]
    in_3: input1 for add [B, 256, 512]
    in_4: input2 for add [B, 256, 512]
    Output: [B, 512, 16, 16]
    """
    B = in_3.shape[0]
    M = B * 256
    N = 512
    
    # Flatten inputs
    in_3_flat = in_3.reshape(M, N).contiguous()
    in_4_flat = in_4.reshape(M, N).contiguous()
    
    out = torch.empty((B, 512, 16, 16), dtype=in_3.dtype, device=in_3.device)
    
    # Launch kernel - one program per row
    grid = (M,)
    
    fused_add_layernorm_reshape_permute_kernel[grid](
        in_3_flat, in_4_flat, in_1, in_0, out,
        M, N,
        1e-05,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    
    return out


def replacement_func():
    return fused_add_layernorm_reshape_permute_variant2