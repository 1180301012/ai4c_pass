import torch
import triton
import triton.language as tl

def pattern(input):
    norm = torch.norm(input, p=2, dim=-1, keepdim=True)
    return input / norm

def replacement_args(input):
    return (input,)

@triton.jit
def norm_div_kernel(
    input_ptr,
    output_ptr,
    n_elements_per_vector,
    n_vectors,
    BLOCK_SIZE: tl.constexpr
):
    vector_id = tl.program_id(0)
    start_idx = vector_id * n_elements_per_vector
    thread_id = tl.thread_id(0)
    if thread_id < n_elements_per_vector:
        x = tl.load(input_ptr + start_idx + thread_id)
        x_sq = x * x
        sum_sq = tl.sum(x_sq, axis=0)
        norm = tl.sqrt(sum_sq)
        y = x / norm
        tl.store(output_ptr + start_idx + thread_id, y)

@torch.fx.wrap
def norm_div(input):
    n_vectors = input.shape[0]
    N = input.shape[-1]
    out = torch.empty_like(input)
    num_blocks = n_vectors
    block_size = 512

    norm_div_kernel[(num_blocks,)](
        input_ptr=input,
        output_ptr=out,
        n_elements_per_vector=N,
        n_vectors=n_vectors,
        BLOCK_SIZE=block_size
    )
    return out

def replacement_func():
    return norm_div