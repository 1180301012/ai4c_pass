import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_1, in_2):
    return torch.nn.functional.embedding(in_1, in_2, None, None, 2.0, False, False)

# Argument extraction function
def replacement_args(in_1, in_2):
    return (in_1, in_2)

# Triton kernel
@triton.jit
def embedding_kernel(
    input_ids_ptr,
    weight_ptr,
    output_ptr,
    B, S, H, V,
    BLOCK_SIZE: tl.constexpr,
):
    bid = tl.program_id(0)
    batch_idx = bid // S
    seq_idx = bid % S

    idx = tl.load(input_ids_ptr + batch_idx * S + seq_idx)
    
    weight_row_ptr = weight_ptr + idx * H
    
    thread_id = tl.thread_id(0)
    start = thread_id * BLOCK_SIZE
    mask = start + tl.arange(0, BLOCK_SIZE) < H
    
    weights = tl.load(weight_row_ptr + start, mask=mask, other=0.0)
    
    output_row_ptr = output_ptr + batch_idx * S * H + seq_idx * H
    tl.store(output_row_ptr + start, weights, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def triton_embedding_wrapper(x, weight):
    B, S = x.shape
    H = weight.shape[1]
    V = weight.shape[0]
    
    out = torch.empty((B, S, H), dtype=weight.dtype, device=weight.device)
    
    num_blocks = B * S
    BLOCK_SIZE = 128
    
    embedding_kernel[(num_blocks,)](
        x,
        weight,
        out,
        B, S, H, V,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return triton_embedding_wrapper