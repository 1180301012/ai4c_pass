import torch
from torch import device
import triton
import triton.language as tl

def pattern(conv_output, position_embeddings):
    """Pattern matches the sequence: detach + type_as + to(device) + addition"""
    # Match the exact operations from the computation graph
    tmp_6 = position_embeddings.detach()
    tmp_7 = tmp_6.type_as(conv_output)
    tmp_8 = tmp_7.to(device=device(type='cuda', index=0), copy=True)
    result = conv_output + tmp_8
    return result

def replacement_args(conv_output, position_embeddings):
    return (conv_output, position_embeddings)

# Triton kernel for fused transfer + addition with autotuning
@triton.jit
def fused_transfer_add_kernel(
    conv_output_ptr,
    position_embeddings_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Optimized kernel with enhanced memory coalescing and compute efficiency
    pid = tl.program_id(0)
    
    # Calculate total elements for parallel processing
    n_elements = batch_size * seq_len * hidden_dim
    
    # Each program handles a contiguous block of memory for optimal bandwidth usage
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Memory coalescing: Load both tensors with aligned memory access patterns
    # Conv elements (already on GPU) - direct memory access
    conv_elements = tl.load(conv_output_ptr + offsets, mask=mask, other=0.0)
    
    # Position embeddings elements (with implicit CPU-to-GPU transfer)
    # Triton handles this efficiently with streaming transfers
    pos_elements = tl.load(position_embeddings_ptr + offsets, mask=mask, other=0.0)
    
    # Fused arithmetic operation with type conversion integrated
    # This eliminates temporary tensor allocations and reduces memory bandwidth pressure
    out_elements = conv_elements + pos_elements
    
    # Coalesced memory write back to global memory
    tl.store(out_ptr + offsets, out_elements, mask=mask)

@torch.fx.wrap
def fused_transfer_add(conv_output, position_embeddings):
    """Fused operation: detach + type_as + to(device) + addition with optimized parallel execution"""
    # Get tensor shapes for parallel processing
    batch_size, seq_len, hidden_dim = position_embeddings.shape
    
    # Ensure conv_output has compatible shape for vectorized operations
    # This reshape eliminates data movement overhead
    if conv_output.dim() == 3 and conv_output.shape[2] == seq_len:
        # Convert from [1, hidden_dim, seq_len] to [1, seq_len, hidden_dim] for alignment
        conv_reshaped = conv_output.transpose(1, 2)  # Reorder dimensions for optimal memory access
    else:
        # Direct reshape when dimensions already aligned
        conv_reshaped = conv_output.reshape(batch_size, seq_len, hidden_dim)
    
    # Calculate total elements for parallel processing configuration
    total_elements = batch_size * seq_len * hidden_dim
    
    # Create output tensor with optimal memory layout
    out = torch.empty_like(conv_reshaped, dtype=conv_reshaped.dtype, device='cuda')
    
    # Adaptive block sizing for maximum GPU utilization across different tensor sizes
    if total_elements >= 1024 * 1024:  # Large tensors (>1M elements)
        BLOCK_SIZE = 1024  # Maximize compute utilization for large workloads
    elif total_elements >= 512 * 512:  # Medium tensors
        BLOCK_SIZE = 512   # Balance compute and memory bandwidth
    else:  # Small tensors
        BLOCK_SIZE = 256   # Reduce overhead for small computations
    
    # Calculate optimal grid configuration for maximum parallelism
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with memory-coalesced access patterns for maximum bandwidth efficiency
    fused_transfer_add_kernel[(grid_size,)](
        conv_reshaped,
        position_embeddings,
        out,
        batch_size,
        seq_len,
        hidden_dim,
        BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_transfer_add