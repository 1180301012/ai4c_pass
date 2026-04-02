import torch
import triton
import triton.language as tl


def pattern(indices, weight):
    return torch.nn.functional.embedding(indices, weight, None, None, 2.0, False, False)


def replacement_args(indices, weight):
    return (indices, weight)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_DIM': 2048}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_DIM': 2048}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_DIM': 2048}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_DIM': 2048}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_DIM': 2048}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_DIM': 1024}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_DIM': 1024}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_DIM': 1024}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_DIM': 512},  num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_DIM': 512},  num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_DIM': 512},  num_warps=16, num_stages=1),
        triton.Config({'BLOCK_DIM': 256},  num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_DIM': 256},  num_warps=8,  num_stages=1),
    ],
    key=['embed_dim'],
)
@triton.jit
def triton_embedding_kernel(
    indices_ptr,
    weight_ptr,
    output_ptr,
    num_tokens,
    embed_dim,
    BLOCK_DIM: tl.constexpr,
):
    token_id = tl.program_id(0)
    block_id = tl.program_id(1)

    idx = tl.load(indices_ptr + token_id)

    start = block_id * BLOCK_DIM
    dim_offsets = start + tl.arange(0, BLOCK_DIM)
    mask = dim_offsets < embed_dim

    # evict_last: keep frequently-reused embedding rows in L1+L2 cache
    # (common tokens like padding appear many times per batch)
    values = tl.load(
        weight_ptr + idx * embed_dim + dim_offsets,
        mask=mask, other=0.0,
        eviction_policy='evict_last',
    )
    # evict_first: output is write-once, don't pollute cache
    tl.store(
        output_ptr + token_id * embed_dim + dim_offsets,
        values, mask=mask,
        eviction_policy='evict_first',
    )


@torch.fx.wrap
def triton_embedding(indices, weight):
    num_tokens = indices.numel()
    embed_dim = weight.shape[1]

    # For small inputs use index-gather (equiv. for forward inference:
    # no padding_idx, no max_norm).  Triton has unavoidable launch overhead
    # that makes it slower than PyTorch's gather kernel for small batches.
    if num_tokens <= 4096:
        return weight[indices]

    # Large-batch path: autotuned Triton kernel.
    orig_shape = indices.shape
    indices_flat = indices.view(-1)
    output = torch.empty(num_tokens, embed_dim,
                         dtype=weight.dtype, device=weight.device)

    grid = lambda meta: (num_tokens, triton.cdiv(embed_dim, meta['BLOCK_DIM']))
    triton_embedding_kernel[grid](
        indices_flat, weight, output, num_tokens, embed_dim,
    )

    return output.reshape(list(orig_shape) + [embed_dim])


def replacement_func():
    return triton_embedding