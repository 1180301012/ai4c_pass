import torch
import triton
import triton.language as tl

# Pattern matching for the computation: (in_5 / in_4).to(torch.float32) + embedding
def pattern(in_5, in_4, in_6, tmp_1):
    tmp_4 = in_5 / in_4
    tmp_5 = tmp_4.to(torch.float32)
    tmp_4 = None
    tmp_6 = torch.nn.functional.embedding(in_6, tmp_1, 1, None, 2.0, False, False)
    tmp_1 = None
    tmp_7 = tmp_5 + tmp_6
    tmp_5 = tmp_6 = None
    return tmp_7

# Argument extraction for the fused pattern
def replacement_args(in_5, in_4, in_6, tmp_1):
    return (in_5, in_4, in_6, tmp_1)

# Triton kernel for fused elementwise operations + embedding
@triton.jit
def fused_elementwise_embedding_kernel(
    input_ptr,
    divisor_ptr,
    position_ids_ptr,
    weight_ptr,
    output_ptr,
    n_elements,
    num_embeddings,
    embedding_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program works on a contiguous block of the output
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load divisor (scalar broadcast for all elements in the block)
    divisor = tl.load(divisor_ptr)
    
    # Load input values
    inputs = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Perform division: input / divisor
    # In Triton, operations are already float32, so no explicit conversion needed
    result = inputs / divisor
    
    # For embedding lookup, we need to handle it separately with a different approach
    # Since embedding requires position IDs, we'll handle it in the wrapper function
    
    # Store the intermediate result
    tl.store(output_ptr + offsets, result, mask=mask)

@triton.jit
def embedding_kernel(
    position_ids_ptr,
    weight_ptr,
    embedding_output_ptr,
    seq_len,
    batch_size,
    embedding_dim,
    num_embeddings,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * seq_len * embedding_dim
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create flattened indices for batch-major order
    # We need to compute: batch_idx, seq_idx, embed_idx from linear index
    total_seq = batch_size * seq_len
    
    batch_idx = offsets // total_seq
    seq_idx = (offsets % total_seq) // seq_len
    embed_idx = offsets % embedding_dim
    
    mask = offsets < total_elements
    
    # Load position ID for this batch and sequence position
    pos_id = tl.load(position_ids_ptr + batch_idx * seq_len + seq_idx)
    
    # Compute embedding weight offset: pos_id * embedding_dim + embed_idx
    weight_offset = pos_id * embedding_dim + embed_idx
    
    # Load embedding weight
    embedding_value = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
    
    # Store result
    tl.store(embedding_output_ptr + offsets, embedding_value, mask=mask)

@torch.fx.wrap
def fused_elementwise_embedding_function(input_tensor, divisor_tensor, position_ids, embedding_weights):
    batch_size, seq_len, hidden_dim = input_tensor.shape
    
    # Number of elements in the input tensor
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    elementwise_output = torch.empty_like(input_tensor, dtype=torch.float32)
    embedding_output = torch.empty((batch_size, seq_len, embedding_weights.shape[1]), dtype=torch.float32, device=input_tensor.device)
    
    # Convert divisor to float for division
    divisor_float = divisor_tensor.float()
    
    # Launch elementwise kernel
    fused_elementwise_embedding_kernel[(num_programs,)](
        input_ptr=input_tensor,
        divisor_ptr=divisor_float,
        position_ids_ptr=position_ids,
        weight_ptr=embedding_weights,
        output_ptr=elementwise_output,
        n_elements=n_elements,
        num_embeddings=embedding_weights.shape[0],
        embedding_dim=embedding_weights.shape[1],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Launch embedding kernel separately (can't be easily fused in the same kernel due to different computation patterns)
    embedding_n_elements = batch_size * seq_len * embedding_weights.shape[1]
    embedding_num_programs = (embedding_n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    embedding_kernel[(embedding_num_programs,)](
        position_ids_ptr=position_ids,
        weight_ptr=embedding_weights,
        embedding_output_ptr=embedding_output,
        seq_len=seq_len,
        batch_size=batch_size,
        embedding_dim=embedding_weights.shape[1],
        num_embeddings=embedding_weights.shape[0],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Addition of the two results
    final_result = elementwise_output + embedding_output
    
    return final_result

def replacement_func():
    return fused_elementwise_embedding_function