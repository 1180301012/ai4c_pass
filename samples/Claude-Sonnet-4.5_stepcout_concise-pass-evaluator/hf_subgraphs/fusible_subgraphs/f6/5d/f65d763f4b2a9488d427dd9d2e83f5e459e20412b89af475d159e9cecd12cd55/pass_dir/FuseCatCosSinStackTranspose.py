import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Pattern to match the computation:
    - Concatenate in_0 and in_2 along last dim
    - Compute cos and sin of in_1, then concatenate
    - Stack the two concatenations
    - Transpose last two dimensions
    """
    tmp_0 = torch.cat((in_0, in_2), dim=-1)
    tmp_1 = in_1.cos()
    tmp_2 = in_1.sin()
    tmp_3 = torch.cat((tmp_1, tmp_2), dim=-1)
    tmp_4 = torch.stack((tmp_0, tmp_3), dim=-1)
    tmp_5 = tmp_4.transpose(-1, -2)
    return tmp_5


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_cat_cossin_stack_transpose_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    n_elements,
    feat_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simplified 1D kernel optimized for coalesced memory access.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Each input has n_elements = batch_size * feat_size elements
    # Output has batch_size * 2 * 2*feat_size elements
    
    # Map to input coordinates
    batch_idx = offsets // feat_size
    feat_idx = offsets % feat_size
    
    # Load all three inputs once
    in_0_val = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1_val = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    in_2_val = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    
    # Compute cos and sin once
    cos_val = tl.cos(in_1_val)
    sin_val = tl.sin(in_1_val)
    
    # Output tensor shape: [batch, 2, 2*feat]
    # We write 4 values per input element:
    # out[batch, 0, feat] = in_0[batch, feat]
    # out[batch, 0, feat+feat_size] = in_2[batch, feat]
    # out[batch, 1, feat] = cos(in_1[batch, feat])
    # out[batch, 1, feat+feat_size] = sin(in_1[batch, feat])
    
    two_feat = 2 * feat_size
    base_out_idx = batch_idx * 2 * two_feat
    
    # Write 4 outputs
    out_idx_0 = base_out_idx + 0 * two_feat + feat_idx
    out_idx_1 = base_out_idx + 0 * two_feat + feat_idx + feat_size
    out_idx_2 = base_out_idx + 1 * two_feat + feat_idx
    out_idx_3 = base_out_idx + 1 * two_feat + feat_idx + feat_size
    
    tl.store(out_ptr + out_idx_0, in_0_val, mask=mask)
    tl.store(out_ptr + out_idx_1, in_2_val, mask=mask)
    tl.store(out_ptr + out_idx_2, cos_val, mask=mask)
    tl.store(out_ptr + out_idx_3, sin_val, mask=mask)


@torch.fx.wrap
def fused_cat_cossin_stack_transpose(in_0, in_1, in_2):
    """
    Wrapper function that launches the fused kernel.
    Inputs: in_0, in_1, in_2 with shape [batch_size, feat_size]
    Output: [batch_size, 2, 2*feat_size]
    """
    batch_size, feat_size = in_0.shape
    n_elements = batch_size * feat_size
    
    # Output shape after all operations
    out = torch.empty((batch_size, 2, 2 * feat_size), device=in_0.device, dtype=in_0.dtype)
    
    # Launch kernel with autotuning
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_cat_cossin_stack_transpose_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_ptr=out,
        n_elements=n_elements,
        feat_size=feat_size,
    )
    
    return out


def replacement_func():
    return fused_cat_cossin_stack_transpose