import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    indices_ptr,
    weight_ptr,
    output_ptr,
    N,
    E: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized embedding lookup kernel using Triton.
    Uses row-major access pattern for weight table.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    indices_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = indices_offsets < N

    # Load indices
    indices = tl.load(indices_ptr + indices_offsets, mask=mask, other=0)

    # For each index in the block, load the entire embedding vector
    # Weight is stored as [num_embeddings, E] in row-major format
    # So weight[index] starts at weight_ptr + index * E
    for j in range(E):
        # Row-major: weight_ptr[index, j] = weight_ptr + index * E + j
        weight_row_ptr = weight_ptr + indices * E + j
        embeddings = tl.load(weight_row_ptr, mask=mask, other=0.0)
        
        output_offsets = indices_offsets * E + j
        tl.store(output_ptr + output_offsets, embeddings, mask=mask)


@torch.fx.wrap
def triton_embedding(indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    """Optimized embedding lookup using Triton."""
    N = indices.numel()
    E = weight.shape[1]
    indices_flat = indices.flatten()
    output = torch.empty((N, E), dtype=weight.dtype, device=weight.device)
    
    # Use a fixed block size - 256 works well for most cases
    BLOCK_SIZE = 256
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    embedding_kernel[(num_programs,)](
        indices_ptr=indices_flat,
        weight_ptr=weight,
        output_ptr=output,
        N=N,
        E=E,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    if indices.dim() == 2:
        output = output.view(indices.shape[0], indices.shape[1], E)
    
    return output


def pattern(ids, weight):
    """Pattern matching embedding lookup."""
    return torch.nn.functional.embedding(ids, weight, 0, None, 2.0, False, False)


def replacement_args(ids, weight):
    return (ids, weight, 0, None, 2.0, False, False)


def replacement_func():
    return triton_embedding