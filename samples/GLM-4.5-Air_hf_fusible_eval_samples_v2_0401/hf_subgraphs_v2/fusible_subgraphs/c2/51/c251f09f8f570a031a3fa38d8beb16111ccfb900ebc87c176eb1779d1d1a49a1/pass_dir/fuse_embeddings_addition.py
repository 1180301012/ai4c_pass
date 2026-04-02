import torch
import triton
import triton.language as tl

def pattern(emb_1, emb_2, emb_3, emb_4, emb_5, emb_6, emb_7, emb_8, emb_9):
    """
    Pattern matching for the sequential addition of embeddings.
    This matches the computational bottleneck where multiple embeddings are added sequentially.
    """
    # Sequential addition pattern - this is the computational bottleneck
    result = emb_1 + emb_2 + emb_3 + emb_4 + emb_5 + emb_6 + emb_7 + emb_8 + emb_9
    
    return result

def replacement_args(emb_1, emb_2, emb_3, emb_4, emb_5, emb_6, emb_7, emb_8, emb_9):
    """Extract arguments needed for the fused vector addition kernel."""
    return (emb_1, emb_2, emb_3, emb_4, emb_5, emb_6, emb_7, emb_8, emb_9)

@triton.jit
def fused_vector_addition_kernel(
    emb1_ptr, emb2_ptr, emb3_ptr, emb4_ptr, emb5_ptr, emb6_ptr, emb7_ptr, emb8_ptr, emb9_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for adding 9 embedding vectors together."""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all 9 embeddings
    emb1 = tl.load(emb1_ptr + offsets, mask=mask, other=0.0)
    emb2 = tl.load(emb2_ptr + offsets, mask=mask, other=0.0)
    emb3 = tl.load(emb3_ptr + offsets, mask=mask, other=0.0)
    emb4 = tl.load(emb4_ptr + offsets, mask=mask, other=0.0)
    emb5 = tl.load(emb5_ptr + offsets, mask=mask, other=0.0)
    emb6 = tl.load(emb6_ptr + offsets, mask=mask, other=0.0)
    emb7 = tl.load(emb7_ptr + offsets, mask=mask, other=0.0)
    emb8 = tl.load(emb8_ptr + offsets, mask=mask, other=0.0)
    emb9 = tl.load(emb9_ptr + offsets, mask=mask, other=0.0)
    
    # Add all 9 embeddings together
    result = emb1 + emb2 + emb3 + emb4 + emb5 + emb6 + emb7 + emb8 + emb9
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)

@triton.jit
def optimized_fused_add_kernel(
    ptr0, ptr1, ptr2, ptr3, ptr4, ptr5, ptr6, ptr7, ptr8,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized fused addition kernel using Triton."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all 9 tensors with contiguous memory access
    x0 = tl.load(ptr0 + offsets, mask=mask, other=0.0)
    x1 = tl.load(ptr1 + offsets, mask=mask, other=0.0)
    x2 = tl.load(ptr2 + offsets, mask=mask, other=0.0)
    x3 = tl.load(ptr3 + offsets, mask=mask, other=0.0)
    x4 = tl.load(ptr4 + offsets, mask=mask, other=0.0)
    x5 = tl.load(ptr5 + offsets, mask=mask, other=0.0)
    x6 = tl.load(ptr6 + offsets, mask=mask, other=0.0)
    x7 = tl.load(ptr7 + offsets, mask=mask, other=0.0)
    x8 = tl.load(ptr8 + offsets, mask=mask, other=0.0)
    
    # Fused addition with efficient vectorization
    result = tl.sum(tl.stack([x0, x1, x2, x3, x4, x5, x6, x7, x8]), axis=0)
    
    # Store result with proper vectorization
    tl.store(output_ptr + offsets, result, mask=mask)

@triton.jit
def simple_fused_add_kernel(
    ptr0, ptr1, ptr2, ptr3, ptr4, ptr5, ptr6, ptr7, ptr8,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple fused addition kernel using Triton."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all 9 tensors with contiguous memory access
    x0 = tl.load(ptr0 + offsets, mask=mask, other=0.0)
    x1 = tl.load(ptr1 + offsets, mask=mask, other=0.0)
    x2 = tl.load(ptr2 + offsets, mask=mask, other=0.0)
    x3 = tl.load(ptr3 + offsets, mask=mask, other=0.0)
    x4 = tl.load(ptr4 + offsets, mask=mask, other=0.0)
    x5 = tl.load(ptr5 + offsets, mask=mask, other=0.0)
    x6 = tl.load(ptr6 + offsets, mask=mask, other=0.0)
    x7 = tl.load(ptr7 + offsets, mask=mask, other=0.0)
    x8 = tl.load(ptr8 + offsets, mask=mask, other=0.0)
    
    # Fused addition with efficient vectorization
    result = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
    
    # Store result with proper vectorization
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_vector_addition(emb_1, emb_2, emb_3, emb_4, emb_5, emb_6, emb_7, emb_8, emb_9):
    """Optimized wrapper function for the fused vector addition kernel."""
    # Verify all tensors have the same shape for simplicity
    shape = emb_1.shape
    for i, emb in enumerate([emb_2, emb_3, emb_4, emb_5, emb_6, emb_7, emb_8, emb_9], 1):
        if emb.shape != shape:
            # Fall back to simple torch.add if shapes don't match
            return emb_1 + emb_2 + emb_3 + emb_4 + emb_5 + emb_6 + emb_7 + emb_8 + emb_9
    
    # All tensors have same shape - use optimized Triton kernel
    n_elements = emb_1.numel()
    
    # Flatten all tensors for contiguous memory access
    flat_tensors = [t.contiguous().view(-1) for t in [emb_1, emb_2, emb_3, emb_4, emb_5, emb_6, emb_7, emb_8, emb_9]]
    
    # Create output tensor
    output = torch.empty_like(flat_tensors[0])
    
    # Optimized block size for GPU architecture
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the optimized kernel
    simple_fused_add_kernel[(num_programs,)](
        ptr0=flat_tensors[0],
        ptr1=flat_tensors[1], 
        ptr2=flat_tensors[2],
        ptr3=flat_tensors[3],
        ptr4=flat_tensors[4],
        ptr5=flat_tensors[5],
        ptr6=flat_tensors[6],
        ptr7=flat_tensors[7],
        ptr8=flat_tensors[8],
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to original shape
    return output.view(shape)

def replacement_func():
    """Return the fused vector addition function."""
    return fused_vector_addition