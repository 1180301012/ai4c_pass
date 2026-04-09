import torch
import triton
import triton.language as tl

# Pattern matching for SILU + Split operations
def pattern(in_1):
    tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
    split = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    tmp_3 = split[0]
    tmp_4 = split[1] 
    tmp_5 = split[2]
    return tmp_1, tmp_3, tmp_4, tmp_5

# Extract arguments for the fused operation
def replacement_args(in_1):
    return (in_1,)

# Optimized fused kernel for SILU + Split
@triton.jit
def fused_silu_split_kernel(
    in_ptr,
    out1_ptr,  # tmp_1 (full silu output)
    out3_ptr,  # tmp_3 (first chunk) 
    out4_ptr,  # tmp_4 (second chunk)
    out5_ptr,  # tmp_5 (third chunk)
    batch_size,
    seq_len,
    chunk1_size,
    chunk2_size, 
    chunk3_size,
    total_dims,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * seq_len * total_dims)
    
    # Linear offset to 3D coordinates
    batch_idx = offsets // (seq_len * total_dims)
    linear_idx = offsets % (seq_len * total_dims)
    seq_idx = linear_idx // total_dims
    dim_idx = linear_idx % total_dims
    
    # Load input
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # SILU activation: x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    silu_out = x * sigmoid_x
    
    # Store full SILU output
    tl.store(out1_ptr + offsets, silu_out, mask=mask)
    
    # Store in appropriate chunk based on dimension index
    chunk1_mask = dim_idx < chunk1_size
    chunk2_mask = (dim_idx >= chunk1_size) & (dim_idx < (chunk1_size + chunk2_size))
    chunk3_mask = dim_idx >= (chunk1_size + chunk2_size)
    
    # Calculate output indices for each chunk
    chunk1_idx = batch_idx * seq_len * chunk1_size + seq_idx * chunk1_size + dim_idx
    chunk2_idx = batch_idx * seq_len * chunk2_size + seq_idx * chunk2_size + (dim_idx - chunk1_size)
    chunk3_idx = batch_idx * seq_len * chunk3_size + seq_idx * chunk3_size + (dim_idx - chunk1_size - chunk2_size)
    
    # Store in respective chunk outputs
    tl.store(out3_ptr + chunk1_idx, silu_out, mask=mask & chunk1_mask)
    tl.store(out4_ptr + chunk2_idx, silu_out, mask=mask & chunk2_mask)
    tl.store(out5_ptr + chunk3_idx, silu_out, mask=mask & chunk3_mask)

# Wrapper function
@torch.fx.wrap  
def fused_silu_split(in_1):
    batch_size, seq_len, total_dims = in_1.shape
    chunk1_size, chunk2_size, chunk3_size = 512, 512, 128
    total_elements = batch_size * seq_len * total_dims
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensors
    tmp_1 = torch.empty_like(in_1)  # Full SILU output
    tmp_3 = torch.empty(batch_size, seq_len, chunk1_size, dtype=in_1.dtype, device=in_1.device)
    tmp_4 = torch.empty(batch_size, seq_len, chunk2_size, dtype=in_1.dtype, device=in_1.device)
    tmp_5 = torch.empty(batch_size, seq_len, chunk3_size, dtype=in_1.dtype, device=in_1.device)
    
    # Launch kernel
    fused_silu_split_kernel[(num_programs,)](
        in_ptr=in_1,
        out1_ptr=tmp_1,
        out3_ptr=tmp_3,
        out4_ptr=tmp_4,
        out5_ptr=tmp_5,
        batch_size=batch_size,
        seq_len=seq_len,
        chunk1_size=chunk1_size,
        chunk2_size=chunk2_size,
        chunk3_size=chunk3_size,
        total_dims=total_dims,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return tmp_1, tmp_3, tmp_4, tmp_5

def replacement_func():
    return fused_silu_split