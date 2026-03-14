import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching for: cat -> reshape -> transpose -> multiply -> pad
    """
    tmp_0 = torch.cat([in_0, in_1, in_2], dim=1)
    tmp_1 = tmp_0.reshape(1, 8, -1, -1)  # Generic reshape pattern
    tmp_2 = tmp_1.transpose(-1, -2)
    tmp_3 = in_3 * tmp_2
    tmp_4 = torch.nn.functional.pad(tmp_3, (0, 0, 1, 0, 0, 0), 'constant', None)
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_cat_reshape_transpose_mul_pad_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, out_ptr,
    # Input shapes
    in_0_c, in_1_c, in_2_c,  # channels for each input
    spatial_size,  # flattened spatial dimension (H*W)
    # Output shape after transpose
    batch_size, num_heads, M, N,  # M is seq_len, N is head_dim
    # Strides
    stride_in_0_b, stride_in_0_c, stride_in_0_s,
    stride_in_1_b, stride_in_1_c, stride_in_1_s,
    stride_in_2_b, stride_in_2_c, stride_in_2_s,
    stride_in_3_b, stride_in_3_h, stride_in_3_m, stride_in_3_n,
    stride_out_b, stride_out_h, stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Get program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Mask for valid indices
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # The output tensor is [batch_size, num_heads, M+1, N] due to padding
    # We need to handle two cases:
    # 1. offs_m[i] == 0: write 0 (padded row)
    # 2. offs_m[i] > 0: compute from inputs
    
    # Initialize output
    out = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Only compute for rows > 0 (row 0 is padding)
    compute_mask = (offs_m > 0) & mask_m
    
    # For rows that need computation, map back to input coordinates
    # After transpose: output[b, h, m, n] comes from transposed[b, h, m, n]
    # Before transpose: transposed[b, h, m, n] = reshaped[b, h, n, m]
    # So we need to read reshaped[b, h, offs_n, offs_m-1] (subtract 1 for padding)
    
    # reshaped is [batch_size, num_heads, N, M]
    # This comes from cat_result reshaped as [1, num_heads, N, M]
    # cat_result is [1, total_channels, spatial_size]
    # where total_channels = in_0_c + in_1_c + in_2_c = num_heads * N
    
    # Map (h, n) to channel index in concatenated tensor
    # channel_idx = h * N + n
    
    for i in range(BLOCK_M):
        m_idx = pid_m * BLOCK_M + i
        if m_idx >= M:
            continue
        
        # Skip padding row
        if m_idx == 0:
            continue
            
        # Adjust for padding (output row m_idx corresponds to data row m_idx-1)
        data_row = m_idx - 1
        
        for j in range(BLOCK_N):
            n_idx = pid_n * BLOCK_N + j
            if n_idx >= N:
                continue
            
            # Map to concatenated tensor coordinates
            # Before reshape: cat_result[0, channel_idx, spatial_idx]
            # where channel_idx = h * N + n, spatial_idx = data_row
            channel_idx = pid_h * N + n_idx
            spatial_idx = data_row
            
            # Determine which input tensor this channel belongs to
            if channel_idx < in_0_c:
                # From in_0
                ptr = in_0_ptr + channel_idx * stride_in_0_c + spatial_idx * stride_in_0_s
                val_cat = tl.load(ptr, mask=spatial_idx < spatial_size, other=0.0)
            elif channel_idx < in_0_c + in_1_c:
                # From in_1
                c_idx = channel_idx - in_0_c
                ptr = in_1_ptr + c_idx * stride_in_1_c + spatial_idx * stride_in_1_s
                val_cat = tl.load(ptr, mask=spatial_idx < spatial_size, other=0.0)
            else:
                # From in_2
                c_idx = channel_idx - in_0_c - in_1_c
                ptr = in_2_ptr + c_idx * stride_in_2_c + spatial_idx * stride_in_2_s
                val_cat = tl.load(ptr, mask=spatial_idx < spatial_size, other=0.0)
            
            # Load from in_3 (no padding offset needed for in_3)
            in_3_ptr_offset = (pid_h * stride_in_3_h + 
                             data_row * stride_in_3_m + 
                             n_idx * stride_in_3_n)
            val_in_3 = tl.load(in_3_ptr + in_3_ptr_offset, mask=data_row < (M-1) and n_idx < N, other=0.0)
            
            # Multiply
            out_val = val_cat * val_in_3
            
            # Store to output (at position m_idx which includes padding)
            out_ptr_offset = (pid_h * stride_out_h + 
                            m_idx * stride_out_m + 
                            n_idx * stride_out_n)
            tl.store(out_ptr + out_ptr_offset, out_val, mask=m_idx < M and n_idx < N)


@torch.fx.wrap
def fused_cat_reshape_transpose_mul_pad(in_0, in_1, in_2, in_3):
    # Get input shapes
    batch_size = in_0.shape[0]
    in_0_c = in_0.shape[1]
    in_1_c = in_1.shape[1]
    in_2_c = in_2.shape[1]
    spatial_size = in_0.shape[2] * in_0.shape[3]
    
    # Compute concatenated tensor shape
    total_channels = in_0_c + in_1_c + in_2_c
    
    # Get output shape from in_3
    num_heads = in_3.shape[1]
    M_before_pad = in_3.shape[2]  # sequence length
    N = in_3.shape[3]  # head dimension
    
    # Output shape after padding: [batch_size, num_heads, M+1, N]
    M = M_before_pad + 1
    out = torch.zeros((batch_size, num_heads, M, N), device=in_0.device, dtype=in_0.dtype)
    
    # Define block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    
    # Launch grid
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
        num_heads,
    )
    
    # Flatten inputs for easier indexing
    in_0_flat = in_0.reshape(batch_size, in_0_c, spatial_size)
    in_1_flat = in_1.reshape(batch_size, in_1_c, spatial_size)
    in_2_flat = in_2.reshape(batch_size, in_2_c, spatial_size)
    
    fused_cat_reshape_transpose_mul_pad_kernel[grid](
        in_0_flat, in_1_flat, in_2_flat, in_3, out,
        in_0_c, in_1_c, in_2_c,
        spatial_size,
        batch_size, num_heads, M, N,
        in_0_flat.stride(0), in_0_flat.stride(1), in_0_flat.stride(2),
        in_1_flat.stride(0), in_1_flat.stride(1), in_1_flat.stride(2),
        in_2_flat.stride(0), in_2_flat.stride(1), in_2_flat.stride(2),
        in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    
    return out


def replacement_func():
    return fused_cat_reshape_transpose_mul_pad