import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
    ],
    key=['B', 'S', 'H'],
)
@triton.jit
def fuse_div_mul_cast_kernel(
    in_0_ptr,  # int64 [B, S]
    in_3_ptr,  # float32 [B, 1, 1]
    in_4_ptr,  # float32 [B, S, H]
    out_ptr,   # output [B, S, H]
    B, S, H,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid: (B, (S*H + BLOCK_SIZE - 1) // BLOCK_SIZE)
    batch_idx = tl.program_id(0)
    seq_feat_start = tl.program_id(1) * BLOCK_SIZE
    
    # Total elements to process
    n_elements = S * H
    
    # Create 1D offsets
    offs = seq_feat_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Compute s and h indices from flattened index
    s_idx = offs // H
    h_idx = offs % H
    
    # Load in_0 (attention mask) - shape [B, S]
    # s_idx can have duplicates, so we need to mask properly
    mask_s = s_idx < S
    in_0 = tl.load(in_0_ptr + batch_idx * S + s_idx, mask=mask_s, other=0.0).to(tl.float32)
    
    # Load in_3 (divisor) - shape [B, 1, 1]
    in_3 = tl.load(in_3_ptr + batch_idx).to(tl.float32)
    
    # Load in_4 - shape [B, S, H]
    in_4 = tl.load(in_4_ptr + batch_idx * S * H + s_idx * H + h_idx, mask=mask, other=0.0).to(tl.float32)
    
    # Compute: (in_4 / in_3) * in_0 (broadcasting in_0 to match in_4)
    # in_0 needs to be broadcast along H dimension
    # We use broadcasting: in_0 is [S], in_4 is [S, H]
    # After broadcasting: in_0 becomes [S, 1]
    result = (in_4 / in_3) * in_0
    
    # Store result
    tl.store(out_ptr + batch_idx * S * H + offs, result, mask=mask)


@torch.fx.wrap
def fuse_div_mul_cast(in_0, in_3, in_4):
    """
    Fused kernel that computes: (in_4 / in_3) * in_0.unsqueeze(-1)
    
    Input shapes:
    - in_0: [B, S] int64 (attention mask)
    - in_3: [B, 1, 1] float32
    - in_4: [B, S, H] float32
    
    Output shape:
    - [B, S, H] float32
    """
    B, S, H = in_4.shape
    out = torch.empty((B, S, H), dtype=torch.float32, device=in_4.device)
    
    # Grid: B batches, and (S*H + BLOCK_SIZE - 1) // BLOCK_SIZE blocks
    n_elements = S * H
    grid = (B, (n_elements + 1024 - 1) // 1024)
    
    fuse_div_mul_cast_kernel[grid](
        in_0, in_3, in_4, out,
        B, S, H,
    )
    
    return out


def pattern(in_0, in_3, in_4):
    """
    Pattern: (in_4 / in_3).to(float32) * in_0.unsqueeze(-1).to(float32)
    Returns the computation result before layer_norm.
    """
    # Division
    tmp_div = in_4 / in_3
    # First cast 
    tmp_cast1 = tmp_div.to(torch.float32)
    # Unsqueeze 
    tmp_unsqueeze = in_0.unsqueeze(-1)
    # Multiplication (tmp_6)
    tmp_mul = tmp_cast1 * tmp_unsqueeze
    # Final cast (tmp_7)
    tmp_result = tmp_mul.to(torch.float32)
    return tmp_result


def replacement_args(in_0, in_3, in_4):
    return (in_0, in_3, in_4)


def replacement_func():
    return fuse_div_mul_cast