import torch
import triton
import triton.language as tl

def pattern(neg_in1, sliced_in3):
    # This matches the pattern: negation + slicing + stacking + reshaping
    # neg_in1 = -in_1  [1, 6, 256, 32]
    # sliced_in3 = in_3[..., slice(None, None, 2)]  [1, 6, 256, 32] 
    # stacked = torch.stack([neg_in1, sliced_in3], dim=-1)  [1, 6, 256, 64]
    # reshaped = stacked.reshape((1, -1, stacked.shape[2], 64))  [1, 6, 256, 64]
    stacked = torch.stack([neg_in1, sliced_in3], dim=-1)
    
    # Get the actual shape and determine reshape parameters dynamically
    if len(stacked.shape) == 4:
        # Keep the last two dimensions, set first to 1, second to the appropriate num_heads
        reshape_shape = (1, stacked.shape[1], stacked.shape[2], 64)
        reshaped = stacked.reshape(reshape_shape)
    else:
        # Fallback for different shapes
        reshaped = stacked
    
    return reshaped

def replacement_args(neg_in1, sliced_in3):
    return (neg_in1, sliced_in3)

@triton.jit
def stack_reshape_kernel(
    neg_in1_ptr,
    sliced_in3_ptr,
    out_ptr,
    batch_size,
    n_heads,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements
    pid = tl.program_id(0)
    total_elements = batch_size * n_heads * seq_len * 64
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate indices 4D layout: [batch, head, seq, feature]
    batch = offsets // (n_heads * seq_len * 64)
    remainder = offsets % (n_heads * seq_len * 64)
    head = remainder // (seq_len * 64)
    remainder = remainder % (seq_len * 64)
    seq = remainder // 64
    feat = remainder % 64
    
    # For first 32 features, load from negated in_1
    first_32_mask = (feat < 32)
    neg_in1_idx = batch * n_heads * seq_len * 32 + head * seq_len * 32 + seq * 32 + feat
    
    # For next 32 features, load from sliced in_3  
    sliced_in3_idx = batch * n_heads * seq_len * 32 + head * seq_len * 32 + seq * 32 + (feat - 32)
    
    # Load data from both sources with proper bounds checking
    neg_in1_val = tl.load(neg_in1_ptr + neg_in1_idx, 
                         mask=first_32_mask & (neg_in1_idx < neg_in1_ptr.shape[0]), 
                         other=0.0)
    sliced_in3_val = tl.load(sliced_in3_ptr + sliced_in3_idx, 
                            mask=~first_32_mask & (sliced_in3_idx < sliced_in3_ptr.shape[0]), 
                            other=0.0)
    
    # Select appropriate value based on feature index (stack operation)
    out_val = tl.where(first_32_mask, neg_in1_val, sliced_in3_val)
    
    # Store result
    tl.store(out_ptr + offsets, out_val, mask=mask)

@torch.fx.wrap
def fused_stack_reshape(neg_in1, sliced_in3):
    batch_size, n_heads, seq_len = neg_in1.shape[:3]
    
    # Calculate optimal block size
    total_elements = batch_size * n_heads * seq_len * 64
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty((batch_size, n_heads, seq_len, 64), dtype=torch.float32, device=neg_in1.device)
    
    # Launch kernel
    stack_reshape_kernel[(num_programs,)](
        neg_in1_ptr=neg_in1,
        sliced_in3_ptr=sliced_in3,
        out_ptr=out,
        batch_size=batch_size,
        n_heads=n_heads,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_stack_reshape