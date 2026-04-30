import torch
import triton
import triton.language as tl


def pattern(ids, weight):
    """Match a single embedding operation."""
    return torch.nn.functional.embedding(ids, weight, 0, None, 2.0, False, False)


def replacement_args(ids, weight):
    return (ids, weight)


# Maximum embedding dimension we support - passed as constexpr
MAX_EMBED_DIM = 2048


@triton.jit
def embedding_kernel(
    output_ptr,
    ids_ptr, weight_ptr,
    batch_size, seq_len, embed_dim,
    ids_batch_stride, ids_seq_stride,
    weight_stride_0, weight_stride_1,
    output_batch_stride, output_seq_stride, output_embed_stride,
    MAX_EMBDIM: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    id_val = tl.load(
        ids_ptr + batch_idx * ids_batch_stride + seq_idx * ids_seq_stride,
        mask=None, eviction_policy="evict_first"
    )
    
    # Use maximum size with mask for variable embed_dim
    offsets = tl.arange(0, MAX_EMBDIM)
    mask = offsets < embed_dim
    
    weight_offset = id_val * weight_stride_0 + offsets * weight_stride_1
    embed = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
    
    output_offset = (
        batch_idx * output_batch_stride + 
        seq_idx * output_seq_stride + 
        offsets * output_embed_stride
    )
    tl.store(output_ptr + output_offset, embed, mask=mask)


@torch.fx.wrap
def triton_embedding(ids, weight):
    batch_size, seq_len = ids.shape
    embed_dim = weight.shape[1]
    
    output = torch.empty((batch_size, seq_len, embed_dim), 
                         dtype=weight.dtype, device=weight.device)
    if output.numel() == 0:
        return output
    
    grid = (batch_size, seq_len)
    
    embedding_kernel[grid](
        output,
        ids, weight,
        batch_size, seq_len, embed_dim,
        ids.stride(0), ids.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        MAX_EMBDIM=MAX_EMBED_DIM,
    )
    
    return output


def replacement_func():
    return triton_embedding