import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    weight_ptr,
    indices_ptr,
    output_ptr,
    padding_idx,
    n_elements,
    num_embeddings,
    embedding_dim,
):
    """
    Optimized embedding lookup kernel.
    Each program handles one embedding lookup.
    """
    pid = tl.program_id(0)
    
    # Bounds check
    if pid >= n_elements:
        return
    
    # Load index as int32 and clamp using min/max (tl.clamp doesn't support integers)
    idx = tl.load(indices_ptr + pid).to(tl.int32)
    idx = tl.minimum(tl.maximum(idx, 0), num_embeddings - 1)
    
    # Check for padding
    is_padding = (idx == padding_idx)
    
    # Base offset in weight matrix
    row_base = idx * embedding_dim
    
    # Load embedding in chunks of 64 elements
    for start in range(0, embedding_dim, 64):
        # Initialize chunk with zeros (bfloat16 to match weight dtype)
        chunk = tl.zeros((64,), dtype=tl.bfloat16)
        
        # Load this chunk
        offsets = row_base + start + tl.arange(0, 64)
        mask = offsets < ((idx + 1) * embedding_dim)
        loaded = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        
        # For first chunk, check padding
        if start == 0:
            padding_vec = tl.full((64,), 2.0, dtype=tl.bfloat16)
            chunk = tl.where(is_padding, padding_vec, loaded)
        else:
            chunk = loaded
        
        # Store
        out_off = pid * embedding_dim + start
        tl.store(output_ptr + out_off + tl.arange(0, 64), chunk, mask=start + tl.arange(0, 64) < embedding_dim)


@torch.fx.wrap
def embedding_wrapper(weight, indices):
    """
    Wrapper for optimized embedding kernel.
    """
    # Get dimensions
    n_elements = indices.numel()
    num_embeddings = weight.shape[0]
    embedding_dim = weight.shape[1]
    
    # Output shape: indices.shape + (embedding_dim,)
    output_shape = list(indices.shape) + [embedding_dim]
    
    # Allocate output
    output = torch.empty(output_shape, dtype=weight.dtype, device=weight.device)
    
    # Launch kernel
    embedding_kernel[(n_elements,)](
        weight,
        indices,
        output,
        -1,  # padding_idx=-1 means no padding
        n_elements,
        num_embeddings,
        embedding_dim,
    )
    
    return output


def pattern(in_1, in_2):
    """
    Pattern matches: torch.nn.functional.embedding(in_1, in_2, None, None, 2.0, False, False)
    """
    tmp_3 = torch.nn.functional.embedding(in_1, in_2, None, None, 2.0, False, False)
    return tmp_3


def replacement_args(in_1, in_2):
    # Swap order: (weight, indices) - in_2 is the embedding table, in_1 is input_ids
    return (in_2, in_1)


def replacement_func():
    return embedding_wrapper