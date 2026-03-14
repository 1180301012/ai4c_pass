import torch
import triton
import triton.language as tl

# Pattern matching function for LayoutLM embedding + independent slice
def pattern(in_0, in_1, in_2):
    """
    Matches embedding lookup and slice operation on independent tensors.
    in_0: embedding weight [vocab_size, embed_dim]
    in_1: indices tensor [batch, seq_len]
    in_2: tensor to slice [batch, seq_len, 4]
    """
    tmp_1 = torch.nn.functional.embedding(in_1, in_0, None, None, 2.0, False, False)
    tmp_2 = in_2[slice(None, None, None), slice(None, None, None), 1]
    return tmp_2, tmp_1

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def embedding_kernel_layoutlm(
    indices_ptr,
    weight_ptr,
    output_ptr,
    num_indices,
    embed_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple embedding kernel - one program per index.
    """
    pid = tl.program_id(0)
    
    if pid >= num_indices:
        return
    
    # Load vocabulary index
    idx = tl.load(indices_ptr + pid)
    
    # Base offsets
    src = idx * embed_dim
    dst = pid * embed_dim
    
    # Copy embedding vector in chunks
    for off in range(0, embed_dim, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < embed_dim
        vals = tl.load(weight_ptr + src + offs, mask=mask)
        tl.store(output_ptr + dst + offs, vals, mask=mask)


@torch.fx.wrap
def triton_embedding_layoutlm(indices, weight):
    """
    Triton embedding lookup for LayoutLM.
    """
    batch_size = indices.shape[0]
    seq_len = indices.shape[1]
    embed_dim = weight.shape[1]
    num_indices = batch_size * seq_len
    
    indices_flat = indices.view(-1)
    emb_output = torch.empty((batch_size, seq_len, embed_dim), dtype=weight.dtype, device=weight.device)
    
    BLOCK_SIZE = 256  # Larger block for embed_dim=768
    grid = (num_indices,)
    
    embedding_kernel_layoutlm[grid](
        indices_flat,
        weight,
        emb_output.view(-1),
        num_indices,
        embed_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return emb_output


def fused_embedding_slice_layoutlm(weight, indices, slice_input):
    """
    Embedding lookup + slice for LayoutLM.
    """
    emb_output = triton_embedding_layoutlm(indices, weight)
    sliced = slice_input[slice(None, None, None), slice(None, None, None), 1]
    return sliced, emb_output


def replacement_func():
    return fused_embedding_slice_layoutlm