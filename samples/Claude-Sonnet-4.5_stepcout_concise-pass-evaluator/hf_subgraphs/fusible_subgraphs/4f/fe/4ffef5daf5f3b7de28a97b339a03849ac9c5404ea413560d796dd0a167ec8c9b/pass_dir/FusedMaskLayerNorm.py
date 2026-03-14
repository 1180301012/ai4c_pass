import torch
import triton
import triton.language as tl


def pattern(mask, norm_factor, input_tensor):
    """
    Pattern to match: mask preprocessing before layer norm
    - Divide input by normalization factor
    - Convert to float32
    - Apply mask (unsqueeze and multiply)
    - Convert to float32 again
    """
    tmp_3 = input_tensor / norm_factor
    tmp_4 = tmp_3.to(torch.float32)
    tmp_5 = mask.unsqueeze(-1)
    tmp_6 = tmp_4 * tmp_5
    tmp_7 = tmp_6.to(torch.float32)
    return tmp_7


def replacement_args(mask, norm_factor, input_tensor):
    return (mask, norm_factor, input_tensor)


@triton.jit
def fused_mask_preprocess_kernel(
    mask_ptr,      # [B, S] - attention mask (int64)
    norm_ptr,      # [B, 1, 1] - normalization factor
    input_ptr,     # [B, S, H] - input tensor
    out_ptr,       # [B, S, H] - output tensor
    B,             # batch size
    S,             # sequence length
    H,             # hidden dimension
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for mask preprocessing:
    - Divide by normalization factor
    - Apply mask (unsqueeze and multiply)
    Each program processes one row (B*S position)
    Optimized for minimal overhead
    """
    # Get the row index (batch * sequence position)
    row_idx = tl.program_id(0)
    b_idx = row_idx // S
    s_idx = row_idx % S
    
    # Load scalar values for this row
    mask_val = tl.load(mask_ptr + b_idx * S + s_idx).to(tl.float32)
    norm_val = tl.load(norm_ptr + b_idx)
    
    # Compute the combined scale factor
    scale = mask_val / norm_val
    
    # Process this row with vectorized operations
    row_start = row_idx * H
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process in chunks
    for chunk in range(0, H, BLOCK_SIZE):
        chunk_offsets = chunk + offsets
        mask = chunk_offsets < H
        
        # Load, scale, and store in one go
        val = tl.load(input_ptr + row_start + chunk_offsets, mask=mask, other=0.0)
        result = val * scale
        tl.store(out_ptr + row_start + chunk_offsets, result, mask=mask)


@torch.fx.wrap
def fused_mask_preprocess(mask, norm_factor, input_tensor):
    """
    Wrapper function for the fused mask preprocessing kernel
    """
    B, S = mask.shape
    H = input_tensor.shape[-1]
    
    # Allocate output
    out = torch.empty_like(input_tensor, dtype=torch.float32)
    
    # Launch kernel with one program per row
    # Use BLOCK_SIZE=512 and num_warps=4 for optimal performance
    grid = (B * S,)
    
    fused_mask_preprocess_kernel[grid](
        mask, norm_factor, input_tensor,
        out,
        B, S, H,
        BLOCK_SIZE=512,
        num_warps=4
    )
    
    return out


def replacement_func():
    return fused_mask_preprocess