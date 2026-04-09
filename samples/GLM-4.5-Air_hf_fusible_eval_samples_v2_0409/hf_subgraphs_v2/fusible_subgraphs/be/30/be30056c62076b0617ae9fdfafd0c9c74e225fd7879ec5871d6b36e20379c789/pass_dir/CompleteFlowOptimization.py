import torch
import triton
import triton.language as tl

# Pattern matching for the complete computation flow
def pattern(in_0, in_1):
    tmp_1 = torch.nn.functional.silu(in_1, inplace=True)  
    split = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    tmp_3 = split[0]
    tmp_4 = split[1]
    tmp_5 = split[2]
    tmp_6 = tmp_5.unsqueeze(2)
    tmp_7 = in_0[(None, None, slice(None, None, None))]
    return tmp_7, tmp_3, tmp_6, tmp_4

# Extract arguments for the complete operation
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel for the complete computation
@triton.jit
def complete_flow_kernel(
    in_0_ptr,
    in_1_ptr,
    out7_ptr,  # tmp_7: expanded in_0
    out3_ptr,  # tmp_3: first chunk
    out6_ptr,  # tmp_6: third chunk with unsqueeze
    out4_ptr,  # tmp_4: second chunk
    in_0_shape,
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
    
    # Process silu + split for in_1
    batch_idx = offsets // (seq_len * total_dims)
    seq_idx = (offsets % (seq_len * total_dims)) // total_dims
    dim_idx = offsets % total_dims
    
    # Load in_1 and apply SILU
    x = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    silu_out = x * sigmoid_x
    
    # Determine chunk and store in appropriate output
    if dim_idx < chunk1_size:
        # First chunk
        chunk_idx = batch_idx * seq_len * chunk1_size + seq_idx * chunk1_size + dim_idx
        tl.store(out3_ptr + chunk_idx, silu_out, mask=mask)
    elif dim_idx < (chunk1_size + chunk2_size):
        # Second chunk  
        chunk_idx = batch_idx * seq_len * chunk2_size + seq_idx * chunk2_size + (dim_idx - chunk1_size)
        tl.store(out4_ptr + chunk_idx, silu_out, mask=mask)
    else:
        # Third chunk
        chunk_idx = batch_idx * seq_len * chunk3_size + seq_idx * chunk3_size + (dim_idx - chunk1_size - chunk2_size)
        tl.store(out6_ptr + chunk_idx, silu_out, mask=mask)
    
    # Process in_0 expansion (broadcasting)
    # This needs separate handling since it has different dimensions
    # For now, leave this to be handled by a separate kernel/optimization

@torch.fx.wrap
def complete_flow_optimized(in_0, in_1):
    batch_size, seq_len, total_dims = in_1.shape
    chunk1_size, chunk2_size, chunk3_size = 512, 512, 128
    
    BLOCK_SIZE = 1024
    num_programs = (batch_size * seq_len * total_dims + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensors
    tmp_3 = torch.empty(batch_size, seq_len, chunk1_size, dtype=in_1.dtype, device=in_1.device)
    tmp_4 = torch.empty(batch_size, seq_len, chunk2_size, dtype=in_1.dtype, device=in_1.device)  
    tmp_5 = torch.empty(batch_size, seq_len, chunk3_size, dtype=in_1.dtype, device=in_1.device)
    tmp_6 = torch.empty(batch_size, seq_len, 1, chunk3_size, dtype=in_1.dtype, device=in_1.device)
    
    # Process silu + split with Triton kernel
    complete_flow_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out7_ptr=None,  # Will be handled separately
        out3_ptr=tmp_3,
        out6_ptr=tmp_5,  # Store third chunk first, then unsqueeze later
        out4_ptr=tmp_4,
        in_0_shape=tl.tensor(in_0.shape, dtype=tl.int32),
        batch_size=batch_size,
        seq_len=seq_len,
        chunk1_size=chunk1_size,
        chunk2_size=chunk2_size,
        chunk3_size=chunk3_size,
        total_dims=total_dims,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Apply unsqueeze to tmp_5 to get tmp_6
    tmp_6 = tmp_5.unsqueeze(2)
    
    # Handle in_0 expansion separately (or with separate kernel)
    tmp_7 = in_0[(None, None, slice(None, None, None))]
    
    return tmp_7, tmp_3, tmp_6, tmp_4

def replacement_func():
    return complete_flow_optimized