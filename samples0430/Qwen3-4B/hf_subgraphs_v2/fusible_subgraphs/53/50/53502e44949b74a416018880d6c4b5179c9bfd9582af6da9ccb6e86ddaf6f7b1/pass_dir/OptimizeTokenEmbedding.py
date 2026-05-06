import torch
import triton
import triton.language as tl

def pattern(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    return torch.nn.functional.embedding(
        input,
        weight,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        sparse
    )

def replacement_args(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    return (input, weight)

@triton.jit
def embed_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_shape,
    weight_shape,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program block handles a contiguous slice of input
    block_id = tl.program_id(0)
    start = block_id * BLOCK_SIZE
    end = min(start + BLOCK_SIZE, input_shape)
    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < (end - start)
    
    # Load input tokens (integer IDs)
    input_ids = tl.load(input_ptr + start, mask=mask, other=-1)
    
    # Initialize output values
    output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # For each token in this block
    for i in tl.arange(0, BLOCK_SIZE):
        if mask[i]:
            token_id = input_ids[i]
            # Calculate the offset in the weight matrix
            weight_offset = token_id * weight_shape
            # Load embedding row
            embedding = tl.load(weight_ptr + weight_offset, mask=tl.arange(0, 1) < 1, other=0.0)
            output[i] = embedding
    
    # Store the output
    tl.store(output_ptr + start, output, mask=mask)

@torch.fx.wrap
def kernel_wrapper(input, weight):
    N = input.numel()
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(input)
    
    embed_kernel[(num_blocks,)](
        input_ptr=input,
        weight_ptr=weight,
        output_ptr=output,
        input_shape=N,
        weight_shape=weight.shape[0],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return kernel_wrapper