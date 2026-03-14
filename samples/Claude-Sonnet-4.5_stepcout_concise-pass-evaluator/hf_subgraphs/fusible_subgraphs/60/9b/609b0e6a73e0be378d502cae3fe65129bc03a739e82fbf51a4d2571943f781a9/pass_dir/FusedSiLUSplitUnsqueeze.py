import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern to match: SiLU + split + unsqueeze + indexing
    """
    tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_2 = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    tmp_3 = tmp_2[0]
    tmp_4 = tmp_2[1]
    tmp_5 = tmp_2[2]
    tmp_6 = tmp_5.unsqueeze(2)
    tmp_7 = in_0[None, None, slice(None, None, None)]
    return (tmp_7, tmp_3, tmp_6, tmp_4)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 256}, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_silu_split_kernel(
    in_ptr,
    out1_ptr,
    out2_ptr,
    out3_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out1_m,
    stride_out1_n,
    stride_out2_m,
    stride_out2_n,
    stride_out3_m,
    stride_out3_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel for SiLU + split
    Input: [B, M, 1152]
    Outputs: 
      - out1: [B, M, 512] (first split)
      - out2: [B, M, 512] (second split)
      - out3: [B, M, 1, 128] (third split with unsqueeze)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate which output this block belongs to
    # out1: 0-511, out2: 512-1023, out3: 1024-1151
    n_offset = pid_n * BLOCK_SIZE_N
    
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = n_offset + tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    
    # Load input data
    in_offsets = m_offsets[:, None] * stride_in_m + n_offsets[None, :] * stride_in_n
    mask = m_mask[:, None] & n_mask[None, :]
    x = tl.load(in_ptr + in_offsets, mask=mask, other=0.0)
    
    # Apply SiLU: x * sigmoid(x)
    sigmoid_x = tl.sigmoid(x)
    silu_x = x * sigmoid_x
    
    # Determine which output to write to based on n_offset
    if n_offset < 512:
        # Write to out1
        out_offsets = m_offsets[:, None] * stride_out1_m + n_offsets[None, :] * stride_out1_n
        tl.store(out1_ptr + out_offsets, silu_x, mask=mask)
    elif n_offset < 1024:
        # Write to out2
        n_out = n_offsets - 512
        n_out_mask = n_out < 512
        out_mask = m_mask[:, None] & n_out_mask[None, :]
        out_offsets = m_offsets[:, None] * stride_out2_m + n_out[None, :] * stride_out2_n
        tl.store(out2_ptr + out_offsets, silu_x, mask=out_mask)
    else:
        # Write to out3 (with unsqueeze at dim=2)
        n_out = n_offsets - 1024
        n_out_mask = n_out < 128
        out_mask = m_mask[:, None] & n_out_mask[None, :]
        out_offsets = m_offsets[:, None] * stride_out3_m + n_out[None, :] * stride_out3_n
        tl.store(out3_ptr + out_offsets, silu_x, mask=out_mask)


@torch.fx.wrap
def fused_silu_split_unsqueeze(in_0, in_1):
    """
    Fused implementation of SiLU + split + unsqueeze
    """
    # Get shapes
    batch_size = in_1.shape[0]
    M = in_1.shape[1]
    N = in_1.shape[2]  # Should be 1152
    
    # Create output tensors
    out1 = torch.empty((batch_size, M, 512), dtype=in_1.dtype, device=in_1.device)
    out2 = torch.empty((batch_size, M, 512), dtype=in_1.dtype, device=in_1.device)
    out3 = torch.empty((batch_size, M, 1, 128), dtype=in_1.dtype, device=in_1.device)
    
    # Handle in_0 transformation (just view operation)
    tmp_7 = in_0[None, None, :]
    
    # Launch kernel for each batch
    BLOCK_SIZE_M = 4
    BLOCK_SIZE_N = 256
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
        batch_size
    )
    
    # Process each batch
    for b in range(batch_size):
        fused_silu_split_kernel[grid](
            in_1[b],
            out1[b],
            out2[b],
            out3[b, :, 0, :],  # Remove the unsqueeze dimension for kernel
            M,
            N,
            in_1.stride(1),
            in_1.stride(2),
            out1.stride(1),
            out1.stride(2),
            out2.stride(1),
            out2.stride(2),
            out3.stride(1),
            out3.stride(3),  # Use stride for the last dimension
        )
    
    return (tmp_7, out1, out3, out2)


def replacement_func():
    return fused_silu_split_unsqueeze