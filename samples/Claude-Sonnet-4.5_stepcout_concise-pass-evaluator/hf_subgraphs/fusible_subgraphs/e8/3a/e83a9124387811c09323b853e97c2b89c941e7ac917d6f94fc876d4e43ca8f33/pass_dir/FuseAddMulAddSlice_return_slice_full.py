import torch
import triton
import triton.language as tl

# Pattern for return (tmp_6, tmp_4) order
def pattern(in_0, in_1, in_2, in_3):
    """Match the pattern: (in_3 + in_2) * in_1 + in_0, then slice [:, 0]"""
    tmp_2 = in_3 + in_2
    tmp_3 = tmp_2 * in_1
    tmp_4 = tmp_3 + in_0
    tmp_6 = tmp_4[slice(None, None, None), 0]
    return (tmp_6, tmp_4)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_D': 128}),
        triton.Config({'BLOCK_SIZE_D': 256}),
        triton.Config({'BLOCK_SIZE_D': 512}),
    ],
    key=['D'],
)
@triton.jit
def fused_add_mul_add_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    in_3_ptr,
    out_ptr,
    B,
    S,
    D,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Fused kernel for (in_3 + in_2) * in_1 + in_0 with broadcasting"""
    # Each program handles one (batch, seq) position
    pid_bs = tl.program_id(0)
    batch_idx = pid_bs // S
    seq_idx = pid_bs % S
    
    # Process all D elements for this (batch, seq) position
    for d_start in range(0, D, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D
        
        # Calculate offset in the 3D tensor [B, S, D]
        base_offset = batch_idx * S * D + seq_idx * D
        offsets_3d = base_offset + d_offsets
        
        # Load from 3D tensors
        in_2 = tl.load(in_2_ptr + offsets_3d, mask=d_mask, other=0.0)
        in_3 = tl.load(in_3_ptr + offsets_3d, mask=d_mask, other=0.0)
        
        # Load from 1D tensors (broadcast)
        in_0 = tl.load(in_0_ptr + d_offsets, mask=d_mask, other=0.0)
        in_1 = tl.load(in_1_ptr + d_offsets, mask=d_mask, other=0.0)
        
        # Compute: (in_3 + in_2) * in_1 + in_0
        tmp = in_3 + in_2
        tmp = tmp * in_1
        result = tmp + in_0
        
        # Store result
        tl.store(out_ptr + offsets_3d, result, mask=d_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_D': 128}),
        triton.Config({'BLOCK_SIZE_D': 256}),
        triton.Config({'BLOCK_SIZE_D': 512}),
    ],
    key=['D'],
)
@triton.jit
def slice_kernel(
    in_ptr,
    out_ptr,
    B,
    S,
    D,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Extract [:, 0, :] slice"""
    # Each program handles one batch position
    pid_b = tl.program_id(0)
    
    # Process all D elements for this batch position
    for d_start in range(0, D, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D
        
        # Source: [B, 0, D]
        src_offset = pid_b * S * D + 0 * D + d_offsets
        # Dest: [B, D]
        dst_offset = pid_b * D + d_offsets
        
        # Load and store
        data = tl.load(in_ptr + src_offset, mask=d_mask, other=0.0)
        tl.store(out_ptr + dst_offset, data, mask=d_mask)

@torch.fx.wrap
def fused_add_mul_add_slice_kernel(in_0, in_1, in_2, in_3):
    """
    Compute: tmp_4 = (in_3 + in_2) * in_1 + in_0
             tmp_6 = tmp_4[:, 0]
    
    in_0, in_1: shape [D] - bias and scale
    in_2, in_3: shape [B, S, D] - input tensors
    """
    # Get shapes
    B, S, D = in_2.shape
    
    # Allocate outputs
    tmp_4 = torch.empty_like(in_2)
    tmp_6 = torch.empty((B, D), dtype=in_2.dtype, device=in_2.device)
    
    # Launch fused kernel for full computation
    grid = (B * S,)
    fused_add_mul_add_kernel[grid](
        in_0, in_1, in_2, in_3, tmp_4,
        B, S, D,
    )
    
    # Launch slice kernel
    grid = (B,)
    slice_kernel[grid](
        tmp_4, tmp_6,
        B, S, D,
    )
    
    return (tmp_6, tmp_4)

def replacement_func():
    return fused_add_mul_add_slice_kernel