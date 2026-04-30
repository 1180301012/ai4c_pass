import torch
import triton
import triton.language as tl


def pattern(in_1, in_2):
    tmp_3 = torch.nn.functional.embedding(in_1, in_2, None, None, 2.0, False, False)
    return tmp_3


def replacement_args(in_1, in_2):
    return (in_1, in_2)


@triton.jit
def embedding_kernel(
    indices_ptr,
    weight_ptr,
    output_ptr,
    EMBEDDING_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid is (num_chunks, num_tokens) so chunks of same token are consecutive in linear order
    block_id = tl.program_id(0)
    token_id = tl.program_id(1)

    # Load index and cast to int32 for faster arithmetic
    idx = tl.load(indices_ptr + token_id).to(tl.int32)

    offset = block_id * BLOCK_SIZE
    cols = offset + tl.arange(0, BLOCK_SIZE)
    mask = cols < EMBEDDING_DIM

    # Read from weight matrix and write to output
    vals = tl.load(weight_ptr + idx * EMBEDDING_DIM + cols, mask=mask)
    tl.store(output_ptr + token_id * EMBEDDING_DIM + cols, vals, mask=mask)


@torch.fx.wrap
def optimized_embedding(in_1, in_2):
    num_tokens = in_1.numel()
    embedding_dim = in_2.shape[1]

    output_shape = list(in_1.shape) + [embedding_dim]
    output = torch.empty(output_shape, dtype=in_2.dtype, device=in_2.device)

    BLOCK_SIZE = 1024
    num_blocks_d = (embedding_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks_d, num_tokens)

    embedding_kernel[grid](
        in_1,
        in_2,
        output,
        EMBEDDING_DIM=embedding_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def replacement_func():
    return optimized_embedding