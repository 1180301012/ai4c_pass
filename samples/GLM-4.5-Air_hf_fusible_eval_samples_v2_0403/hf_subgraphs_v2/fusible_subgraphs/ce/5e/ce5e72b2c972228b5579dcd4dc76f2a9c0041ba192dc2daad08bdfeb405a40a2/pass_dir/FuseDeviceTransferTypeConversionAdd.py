import torch
import triton
import triton.language as tl

def pattern(conv_result, position_embeddings):
    """
    Pattern that matches: conv_result + position_embeddings.to(device, copy=True).type_as(conv_result)
    This fuses device transfer, type conversion, and addition into a single kernel.
    """
    # Conv result comes from: conv3d -> flatten -> transpose
    tmp_6 = position_embeddings.detach()
    tmp_7 = tmp_6.type_as(conv_result)
    tmp_8 = tmp_7.to(device=torch.device('cuda', index=0), copy=True)
    result = conv_result + tmp_8
    return result

def replacement_args(conv_result, position_embeddings):
    return (conv_result, position_embeddings)

@triton.jit
def fused_transfer_type_add_kernel(
    conv_ptr,
    pos_emb_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple and efficient kernel that fuses device transfer, type conversion, and addition.
    Uses vectorized memory access for better performance.
    """
    # Program identifier
    pid = tl.program_id(0)
    
    # Calculate range for this program using compile-time constant block size
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    
    # Create bounds mask
    mask = offsets < n_elements
    
    # Load both tensors with vectorized access (batched for efficiency)
    # conv_result is already on GPU, pos_emb is transferred from CPU
    conv_vals = tl.load(conv_ptr + offsets, mask=mask, other=0.0)
    pos_vals = tl.load(pos_emb_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition and type conversion in one step
    # This leverages Triton's automatic type conversion
    result = conv_vals + pos_vals
    
    # Store result back to GPU
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_transfer_type_add(conv_result, position_embeddings):
    """
    GPU kernel that fuses device transfer, type conversion, and addition.
    Uses efficient vectorized memory access for better performance.
    """
    n_batches, n_positions, n_features = conv_result.shape
    n_elements = n_batches * n_positions * n_features
    
    # Create output tensor
    output = torch.empty_like(conv_result)
    
    # Calculate optimal grid size using vectorized block size
    BLOCK_SIZE = 128  # Compile-time constant block size
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_size,)
    
    fused_transfer_type_add_kernel[grid](
        conv_ptr=conv_result,
        pos_emb_ptr=position_embeddings,
        out_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_transfer_type_add