import torch
import triton
import triton.language as tl

def pattern(mask_input):
    """Pattern to match: attention mask processing operations only"""
    # Focus on the consistent operations: type conversion, subtraction, multiplication
    # This matches the core computation: tmp_1 = in_1.to(dtype=torch.float32)
    # tmp_2 = 1.0 - tmp_1  
    # tmp_3 = tmp_2 * -3.4028234663852886e+38
    tmp_1 = mask_input.to(dtype=torch.float32)
    tmp_2 = 1.0 - tmp_1
    tmp_3 = tmp_2 * -3.4028234663852886e+38
    return tmp_3

def replacement_args(mask_input):
    """Extract arguments for the replacement function"""
    return (mask_input,)

@triton.jit
def attention_mask_kernel(
    mask_ptr,           # input mask tensor (int64)
    out_ptr,            # output processed mask (float32)
    mask_size,          # total number of elements in mask
    LARGE_NEG_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized Triton kernel for attention mask processing"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < mask_size
    
    # Load mask values as float32 (convert from int64) with memory coalescing
    mask_values = tl.load(mask_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Process in one operation: fuse inversion and multiplication for better ILP
    # This creates very large negative values where mask was 0
    processed_mask = (1.0 - mask_values) * LARGE_NEG_CONST
    
    # Store results - ensure coalesced memory access
    tl.store(out_ptr + offsets, processed_mask, mask=mask)

@torch.fx.wrap
def optimized_mask_processing(mask_input):
    """Optimized function that processes attention mask using hybrid PyTorch/Triton approach"""
    mask_size = mask_input.numel()
    
    # Use regular PyTorch for very small tensors where Triton overhead dominates
    if mask_size < 32:
        return (1.0 - mask_input.to(dtype=torch.float32)) * -3.4028234663852886e+38
    
    # Use Triton for larger tensors
    processed_mask = torch.empty_like(mask_input, dtype=torch.float32)
    
    # Select optimal block size based on tensor size
    if mask_size < 128:
        BLOCK_SIZE = 32      # Small tensors - use smaller blocks to reduce overhead
    elif mask_size < 1024:
        BLOCK_SIZE = 64      # Medium tensors
    else:
        BLOCK_SIZE = 128     # Large tensors - use larger blocks
    
    num_programs = (mask_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Only use Triton if we have multiple work items (worth the overhead)
    if num_programs > 1:
        attention_mask_kernel[(num_programs,)](
            mask_ptr=mask_input,
            out_ptr=processed_mask,
            mask_size=mask_size,
            LARGE_NEG_CONST=-3.4028234663852886e+38,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return processed_mask
    else:
        # Fall back to PyTorch for single work item
        return (1.0 - mask_input.to(dtype=torch.float32)) * -3.4028234663852886e+38

def replacement_func():
    """Return the optimized mask processing function"""
    return optimized_mask_processing