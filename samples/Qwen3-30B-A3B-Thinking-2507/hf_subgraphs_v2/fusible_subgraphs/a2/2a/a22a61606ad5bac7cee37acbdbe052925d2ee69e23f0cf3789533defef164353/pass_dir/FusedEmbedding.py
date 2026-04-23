import torch
import triton
import triton.language as tl

def pattern(input_ids, word_embeddings, token_type_ids, token_type_embeddings, position_ids, position_embeddings):
    tmp_7 = torch.nn.functional.embedding(input_ids, word_embeddings, 0, None, 2.0, False, False)
    tmp_8 = torch.nn.functional.embedding(token_type_ids, token_type_embeddings, None, None, 2.0, False, False)
    tmp_9 = tmp_7 + tmp_8
    tmp_10 = torch.nn.functional.embedding(position_ids, position_embeddings, None, None, 2.0, False, False)
    tmp_9 += tmp_10
    return tmp_9

def replacement_args(input_ids, word_embeddings, token_type_ids, token_type_embeddings, position_ids, position_embeddings):
    return (input_ids, word_embeddings, token_type_ids, token_type_embeddings, position_ids, position_embeddings)

@triton.jit
def fused_embedding_kernel(
    input_ids_ptr,
    token_type_ids_ptr,
    position_ids_ptr,
    word_embeddings_ptr,
    token_type_embeddings_ptr,
    position_embeddings_ptr,
    output_ptr,
    batch_size,
    seq_length,
    hidden_size,
    vocab_size,
    num_token_types,
    max_position_embeddings,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * seq_length

    # Load input ids for the current block
    input_ids = tl.load(input_ids_ptr + offsets, mask=mask, other=0)
    token_type_ids = tl.load(token_type_ids_ptr + offsets, mask=mask, other=0)
    position_ids = tl.load(position_ids_ptr + offsets, mask=mask, other=0)

    # For each hidden dimension, process one element at a time
    for i in range(0, hidden_size, BLOCK_SIZE):
        block_hidden = min(BLOCK_SIZE, hidden_size - i)
        hidden_offsets = i + tl.arange(0, block_hidden)
        hidden_mask = hidden_offsets < hidden_size

        # Load word embeddings
        word_embedding = tl.load(
            word_embeddings_ptr + (input_ids * hidden_size + hidden_offsets),
            mask=mask[:, None] & hidden_mask[None, :],
            other=0.0
        )

        # Load token type embeddings
        token_type_embedding = tl.load(
            token_type_embeddings_ptr + (token_type_ids * hidden_size + hidden_offsets),
            mask=mask[:, None] & hidden_mask[None, :],
            other=0.0
        )

        # Load position embeddings
        position_embedding = tl.load(
            position_embeddings_ptr + (position_ids * hidden_size + hidden_offsets),
            mask=mask[:, None] & hidden_mask[None, :],
            other=0.0
        )

        # Add them together
        result = word_embedding + token_type_embedding + position_embedding

        # Store the result
        tl.store(
            output_ptr + (offsets * hidden_size + hidden_offsets),
            result,
            mask=mask[:, None] & hidden_mask[None, :]
        )

@torch.fx.wrap
def fused_embedding_wrapper(input_ids, word_embeddings, token_type_ids, token_type_embeddings, position_ids, position_embeddings):
    batch_size, seq_length = input_ids.shape
    hidden_size = word_embeddings.shape[1]

    # Calculate grid size
    n_elements = batch_size * seq_length
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Allocate output
    output = torch.empty((batch_size, seq_length, hidden_size), dtype=word_embeddings.dtype, device=word_embeddings.device)

    # Launch kernel
    fused_embedding_kernel[
        (num_blocks,)
    ](
        input_ids,
        token_type_ids,
        position_ids,
        word_embeddings,
        token_type_embeddings,
        position_embeddings,
        output,
        batch_size,
        seq_length,
        hidden_size,
        word_embeddings.shape[0],
        token_type_embeddings.shape[0],
        position_embeddings.shape[0],
        BLOCK_SIZE
    )

    return output

def replacement_func():
    return fused_embedding_wrapper