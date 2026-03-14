import torch
import triton
import triton.language as tl

# Pattern to match: add + unsqueeze + multiply + cast
def pattern(x, emb, mask):
    tmp_7 = x + emb
    tmp_8 = mask.unsqueeze(-1)
    tmp_9 = tmp_7 * tmp_8
    tmp_10 = tmp_9.to(torch.float32)
    return tmp_10

def replacement_args(x, emb, mask):
    return (x, emb, mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=2, num_stages=2),
    ],
    key=['hidden_size'],
)
@triton.jit
def fused_add_mul_cast_kernel(
    x_ptr,
    emb_ptr,
    mask_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    batch_idx = row_idx // seq_len
    seq_idx = row_idx % seq_len
    
    row_start = row_idx * hidden_size
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask_col = col_offsets < hidden_size
    
    # Load x and emb
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask_col, other=0.0).to(tl.float32)
    emb = tl.load(emb_ptr + row_start + col_offsets, mask=mask_col, other=0.0).to(tl.float32)
    
    # Load attention mask - shape is [batch, seq], needs to be broadcast
    attn_mask = tl.load(mask_ptr + batch_idx * seq_len + seq_idx).to(tl.float32)
    
    # Fused: out = (x + emb) * mask
    out = (x + emb) * attn_mask
    
    # Store output (already float32)
    tl.store(out_ptr + row_start + col_offsets, out, mask=mask_col)

@torch.fx.wrap
def fused_add_mul_cast(x, emb, mask):
    orig_shape = x.shape  # [batch, seq, hidden]
    batch_size = orig_shape[0]
    seq_len = orig_shape[1]
    hidden_size = orig_shape[2]
    
    # Flatten x and emb to 2D
    x_2d = x.reshape(-1, hidden_size)
    emb_2d = emb.reshape(-1, hidden_size)
    num_rows = x_2d.shape[0]
    
    # Output tensor
    out = torch.empty(x_2d.shape, dtype=torch.float32, device=x.device)
    
    # Launch kernel
    grid = (num_rows,)
    
    fused_add_mul_cast_kernel[grid](
        x_2d, emb_2d, mask, out,
        batch_size, seq_len, hidden_size,
    )
    
    # Reshape output
    out = out.reshape(orig_shape)
    
    return out

def replacement_func():
    return fused_add_mul_cast