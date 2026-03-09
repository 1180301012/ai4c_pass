import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Match permute followed by reshape pattern
    tmp_2 = in_0.permute(0, 2, 1)
    tmp_3 = tmp_2.reshape(32, 64, 128, 128)
    return tmp_3

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_permute_reshape_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    orig_seq_len,
    hidden_dim,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < (batch_size * 64 * 128 * 128)
    
    # Load original data [32, 16384, 64] -> flattened
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Convert flattened offset to original coordinates [batch, seq, hidden_dim]
    total_per_batch = orig_seq_len * hidden_dim
    batch = offsets // total_per_batch
    remainder = offsets % total_per_batch
    orig_seq = remainder // hidden_dim
    orig_hidden = remainder % hidden_dim
    
    # Apply permute(0, 2, 1): [batch, hidden_dim, seq]
    permuted_batch = batch
    permuted_hidden_dim = orig_hidden  # hidden_dim stays in middle position
    permuted_seq = orig_seq  # seq moves from middle to last position
    
    # Apply reshape(32, 64, 128, 128)
    # Final shape: [batch_size, 64, 128, 128]
    # Where 64 = hidden_dim, 128*128 = seq_len
    final_batch = permuted_batch
    final_hidden_dim = permuted_hidden_dim  # position 1
    final_row = permuted_seq // 128  # position 2 (first 128 dim)
    final_col = permuted_seq % 128   # position 3 (second 128 dim)
    
    # Calculate final offset in [32, 64, 128, 128] tensor
    final_offset = (final_batch * 64 * 128 * 128 + 
                   final_hidden_dim * 128 * 128 + 
                   final_row * 128 + 
                   final_col)
    
    # Store directly to target location
    tl.store(out_ptr + final_offset, x, mask=mask)

@torch.fx.wrap
def fused_permute_reshape(in_0, in_1):
    # Input shape: [32, 16384, 64] for in_0
    batch_size, seq_len, hidden_dim = in_0.shape
    total_elements = in_0.numel()
    
    # Target shape after permute + reshape: [batch_size, hidden_dim, 128, 128]
    # Note: seq_len should equal 128 * 128 = 16384
    out_shape = (batch_size, hidden_dim, 128, 128)
    out_total_elements = batch_size * hidden_dim * 128 * 128
    
    block_size = 1024
    num_programs = (out_total_elements + block_size - 1) // block_size
    
    out = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    out_flat = out.flatten()
    
    fused_permute_reshape_kernel[(num_programs,)](
        in_0,
        out_flat,
        batch_size,
        seq_len,
        hidden_dim,
        block_size=block_size,
    )
    
    return out

def replacement_func():
    return fused_permute_reshape