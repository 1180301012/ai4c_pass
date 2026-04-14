import torch
import triton
import triton.language as tl

def pattern(tmp_0, tmp_9, tmp_15, tmp_6, tmp_18, tmp_10, tmp_20, tmp_11, tmp_22, tmp_10_2, tmp_24, tmp_11_2, tmp_28, tmp_5, tmp_32, tmp_8, tmp_1, tmp_7):
    """
    Pattern to match multiple embedding operations that are sequentially added.
    This matches the embedding operations in the transformer model:
    - tmp_16 = embedding(tmp_0, tmp_9)
    - tmp_17 = embedding(tmp_15, tmp_6) 
    - tmp_19 = embedding(tmp_18, tmp_10)
    - tmp_21 = embedding(tmp_20, tmp_11)
    - tmp_23 = embedding(tmp_22, tmp_10_2)
    - tmp_25 = embedding(tmp_24, tmp_11_2)
    - tmp_29 = embedding(tmp_28, tmp_5)
    - tmp_33 = embedding(tmp_32, tmp_8)
    - tmp_34 = embedding(tmp_1, tmp_7)
    
    Then they are added sequentially: tmp_35 = tmp_16 + tmp_17 + tmp_19 + tmp_21 + tmp_23 + tmp_25 + tmp_29 + tmp_33 + tmp_34
    """
    tmp_16 = torch.nn.functional.embedding(tmp_0, tmp_9, 0, None, 2.0, False, False)
    tmp_17 = torch.nn.functional.embedding(tmp_15, tmp_6, None, None, 2.0, False, False)
    tmp_19 = torch.nn.functional.embedding(tmp_18, tmp_10, None, None, 2.0, False, False)
    tmp_21 = torch.nn.functional.embedding(tmp_20, tmp_11, None, None, 2.0, False, False)
    tmp_23 = torch.nn.functional.embedding(tmp_22, tmp_10_2, None, None, 2.0, False, False)
    tmp_25 = torch.nn.functional.embedding(tmp_24, tmp_11_2, None, None, 2.0, False, False)
    tmp_29 = torch.nn.functional.embedding(tmp_28, tmp_5, None, None, 2.0, False, False)
    tmp_33 = torch.nn.functional.embedding(tmp_32, tmp_8, None, None, 2.0, False, False)
    tmp_34 = torch.nn.functional.embedding(tmp_1, tmp_7, None, None, 2.0, False, False)
    
    tmp_35 = tmp_16 + tmp_17
    tmp_36 = tmp_35 + tmp_19
    tmp_37 = tmp_36 + tmp_21
    tmp_38 = tmp_37 + tmp_23
    tmp_39 = tmp_38 + tmp_25
    tmp_40 = tmp_39 + tmp_29
    tmp_41 = tmp_40 + tmp_33
    tmp_42 = tmp_41 + tmp_34
    
    return tmp_16, tmp_17, tmp_19, tmp_21, tmp_23, tmp_25, tmp_29, tmp_33, tmp_34, tmp_42

def replacement_args(tmp_0, tmp_9, tmp_15, tmp_6, tmp_18, tmp_10, tmp_20, tmp_11, tmp_22, tmp_10_2, tmp_24, tmp_11_2, tmp_28, tmp_5, tmp_32, tmp_8, tmp_1, tmp_7):
    return (tmp_0, tmp_9, tmp_15, tmp_6, tmp_18, tmp_10, tmp_20, tmp_11, tmp_22, tmp_10_2, tmp_24, tmp_11_2, tmp_28, tmp_5, tmp_32, tmp_8, tmp_1, tmp_7)

@triton.jit
def fused_embedding_kernel(
    input_ptrs,
    weight_ptrs,
    output_ptr,
    n_elements,
    embedding_dims,
    num_embeddings_list,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused embedding kernel that processes multiple embedding operations"""
    idx = tl.program_id(0)
    offset = idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < embedding_dims[0]  # First dimension is batch_size, seq_len
    
    # Load input indices for the first embedding
    input_idx = tl.load(input_ptrs[0] + offset, mask=mask, other=0).to(tl.int32)
    
    # Initialize output with first embedding
    output = tl.load(weight_ptrs[0] + input_idx * embedding_dims[1], mask=mask, other=0.0)
    
    # Accumulate remaining embeddings
    for i in range(1, len(weight_ptrs)):
        input_idx = tl.load(input_ptrs[i] + offset, mask=mask, other=0).to(tl.int32)
        embedding = tl.load(weight_ptrs[i] + input_idx * embedding_dims[1], mask=mask, other=0.0)
        output = output + embedding
    
    tl.store(output_ptr + offset, output, mask=mask)

@torch.fx.wrap
def fused_embedding_operation(inputs, weights, output_shape):
    """Wrapper for fused embedding operation"""
    batch_size, seq_len = inputs[0].shape[:2]
    embedding_dim = weights[0].shape[1]
    
    output = torch.empty(output_shape, dtype=weights[0].dtype, device=weights[0].device)
    
    # Prepare pointers for all embeddings
    input_ptrs = [inp.contiguous() for inp in inputs]
    weight_ptrs = [w.contiguous() for w in weights]
    
    # Calculate grid size
    n_elements = batch_size * seq_len * embedding_dim
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_embedding_kernel[grid_size](
        input_ptrs,
        weight_ptrs,
        output,
        n_elements,
        (batch_size, seq_len, embedding_dim),
        [w.shape[0] for w in weights],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_embedding_operation