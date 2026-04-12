import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # EXACT replication of model.py computation - no optimizations, just matching
    tmp_1 = in_0 * 1000000.0
    tmp_2 = in_1 - tmp_1
    split = tmp_2.split(1, dim=-1)
    tmp_4 = split[0]
    tmp_5 = split[1]
    tmp_6 = tmp_4.squeeze(-1)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_5.squeeze(-1)
    tmp_9 = tmp_8.contiguous()
    return (tmp_7, tmp_9)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Fusion kernel that handles both float16 and bfloat16
@triton.jit
def fused_kernel(
    # Input pointers and shapes  
    in_0_ptr, in_1_ptr,
    out_0_ptr, out_1_ptr,
    batch_size, total_elements,
    # Type specialization
    target_dtype: tl.constexpr,
    target_scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # For each position, we need to process both elements from the [2] dimension
    # The input has shape [batch_size, seq_len, 2], so we have 2 elements per "logical" position
    base_idx = offsets
    element_0_idx = base_idx * 2      # First element of the pair
    element_1_idx = base_idx * 2 + 1  # Second element of the pair
    
    element_0_mask = element_0_idx < (total_elements * 2)
    element_1_mask = element_1_idx < (total_elements * 2)
    
    # Load the base value (same for both elements in the pair)
    in_0_val = tl.load(in_0_ptr + base_idx, mask=mask, other=0.0)
    
    # Load both from the pair
    in_1_val_0 = tl.load(in_1_ptr + element_0_idx, mask=element_0_mask, other=0.0)
    in_1_val_1 = tl.load(in_1_ptr + element_1_idx, mask=element_1_mask, other=0.0)
    
    # Convert and compute: tmp_1 = in_0 * 1000000.0, tmp_2 = in_1 - tmp_1
    converted_in_0 = tl.cast(in_0_val, target_dtype) * target_scale
    
    result_0 = in_1_val_0 - converted_in_0
    result_1 = in_1_val_1 - converted_in_0
    
    # Store results directly to the output tensors (after squeeze operation)
    out_0_mask = offsets < total_elements
    out_1_mask = offsets < total_elements
    
    tl.store(out_0_ptr + offsets, result_0, mask=out_0_mask)
    tl.store(out_1_ptr + offsets, result_1, mask=out_1_mask)

@torch.fx.wrap
def fused_compute(in_0, in_1):
    # Determine output dtype based on input dtypes
    if in_1.dtype == torch.float16:
        target_dtype = tl.float16
        scale = 1000000.0
    elif in_1.dtype == torch.bfloat16:
        target_dtype = tl.bfloat16
        scale = 1000000.0
    else:
        # Fallback to float32
        target_dtype = tl.float32
        scale = 1000000.0
    
    # Move in_0 to the same device as in_1 if needed
    if in_0.device != in_1.device:
        in_0 = in_0.to(in_1.device)
    
    # Handle dtype conversion for in_0 - it might be int64 or already float
    if in_0.dtype != in_1.dtype:
        # Convert to match in_1's dtype
        in_0 = in_0.to(in_1.dtype)
    
    # Calculate output shapes after squeeze operations
    batch_size, seq_len, _ = in_1.shape
    total_elements = batch_size * seq_len
    
    # Create output tensors (after squeeze operations)
    out_0 = torch.empty((batch_size, seq_len), dtype=in_1.dtype, device=in_1.device)
    out_1 = torch.empty((batch_size, seq_len), dtype=in_1.dtype, device=in_1.device)
    
    # Block size for the kernel
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    fused_kernel[(num_programs,)](
        in_0, in_1, 
        out_0, out_1,
        batch_size, total_elements,
        target_dtype, scale,
        BLOCK_SIZE
    )
    
    return out_0, out_1

def replacement_func():
    return fused_compute