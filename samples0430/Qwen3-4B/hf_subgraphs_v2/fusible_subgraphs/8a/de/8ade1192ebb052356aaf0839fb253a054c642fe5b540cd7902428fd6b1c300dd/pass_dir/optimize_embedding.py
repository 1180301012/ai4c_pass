import torch
import triton
import triton.language as tl

def pattern(indices, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    return torch.nn.functional.embedding(
        indices,
        weight,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        sparse
    )

def replacement_args(indices, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    return (indices, weight)

@triton.jit
def embedding_kernel(
    indices_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    output_ptr: tl.tensor,
    indices_count: tl.int32,
    vocab_size: tl.int32,
    embedding_dim: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE
    
    indices = tl.load(indices_ptr + start, mask=mask, other=0)
    
    for i in range(BLOCK_SIZE):
        if mask[i]:
            idx = indices[i]
            tl.store(output_ptr + start + i, idx)

@torch.fx.wrap
def kernel_wrapper(
    indices: torch.Tensor,
    weight: torch.Tensor,
    padding_idx: int = None,
    max_norm: float = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False
):
    vocab_size = weight.shape[0]
    embedding_dim = weight.shape[1]
    N = indices.numel()
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(indices)
    
    embedding_kernel[(num_blocks,)](
        indices_ptr=indices,
        weight_ptr=weight,
        output_ptr=output,
        indices_count=N,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def replacement_func():
    return kernel_wrapper