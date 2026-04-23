import torch
import triton
import triton.language as tl

def pattern(in_1, in_2):
    return torch.nn.functional.embedding(in_1, in_2, None, None, 2.0, False, False)

def replacement_args(in_1, in_2):
    return (in_1, in_2)

@triton.jit
def embed_kernel(input_ids_ptr, weight_ptr, out_ptr, B: tl.int32, S: tl.int32, D: tl.int32, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (B * S * D)

    output_idx = offsets
    b = output_idx // (S * D)
    r = output_idx % (S * D)
    s = r // D
    d = r % D

    input_id = tl.load(input_ids_ptr + (b * S + s))
    weight_offset = input_id * D + d
    weight_val = tl.load(weight_ptr + weight_offset, mask=mask)
    tl.store(out_ptr + output_idx, weight_val, mask=mask)

@torch.fx.wrap
def embed_wrapper(input_ids, weight):
    B, S = input_ids.shape
    D = weight.shape[1]
    out = torch.empty((B, S, D), dtype=torch.bfloat16, device=weight.device)
    BLOCK_SIZE = 256
    num_blocks = (B * S * D + BLOCK_SIZE - 1) // BLOCK_SIZE
    embed_kernel[(num_blocks,)](input_ids, weight, out, B, S, D, BLOCK_SIZE)
    return out

def replacement_func():
    return embed_wrapper