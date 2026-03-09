import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Match view followed by transpose pattern
    tmp_0 = in_1.view(32, -1, 1, 64)
    tmp_1 = tmp_0.transpose(1, 2)
    return tmp_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_view_transpose_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < (batch_size * seq_len * hidden_dim)
    
    # Load original data [32, 16384, 64] -> flattened
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Convert flattened offset to original coordinates [batch, seq, hidden_dim]
    total_per_batch = seq_len * hidden_dim
    batch = offsets // total_per_batch
    remainder = offsets % total_per_batch
    orig_seq = remainder // hidden_dim
    orig_channel = remainder % hidden_dim
    
    # Apply view(32, -1, 1, 64): [batch, 256, 1, hidden_dim]
    # where 256 = seq_len / hidden_dim (16384 / 64 = 256)
    view_batch = batch
    view_seq_chunk = orig_seq // hidden_dim  # Each chunk has hidden_dim elements
    view_singleton = 1
    view_channel = orig_channel
    
    # Apply transpose(1, 2): [batch, 1, 256, hidden_dim]
    final_batch = view_batch
    final_dim1 = view_singleton  # position 1 becomes singleton
    final_dim2 = view_seq_chunk  # position 2 becomes seq chunk
    final_dim3 = view_channel  # position 3 becomes channel
    
    # Calculate final offset in [batch, 1, 256, hidden_dim] tensor
    # Where hidden_dim = 64, 256 = seq_len / hidden_dim
    final_offset = (final_batch * 1 * 256 * hidden_dim + 
                   final_dim1 * 256 * hidden_dim + 
                   final_dim2 * hidden_dim + 
                   final_dim3)
    
    # Store directly to target location
    tl.store(out_ptr + final_offset, x, mask=mask)

@torch.fx.wrap
def fused_view_transpose(in_0, in_1):
    # Input shape: [32, 16384, 64] for in_1
    batch_size, seq_len, hidden_dim = in_1.shape
    total_elements = in_1.numel()
    
    # Target shape after view + transpose: [batch_size, 1, seq_len//hidden_dim, hidden_dim]
    out_shape = (batch_size, 1, seq_len // hidden_dim, hidden_dim)
    out_total_elements = batch_size * 1 * (seq_len // hidden_dim) * hidden_dim
    
    block_size = 1024
    num_programs = (out_total_elements + block_size - 1) // block_size
    
    out = torch.empty(out_shape, dtype=in_1.dtype, device=in_1.device)
    out_flat = out.flatten()
    
    fused_view_transpose_kernel[(num_programs,)](
        in_1,
        out_flat,
        batch_size,
        seq_len,
        hidden_dim,
        block_size=block_size,
    )
    
    return out

def replacement_func():
    return fused_view_transpose