import torch
from torch import device
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_1 = in_1.to(device(type='cuda', index=0))
    tmp_2 = torch.nn.functional.embedding(tmp_1, in_0, None, None, 2.0, False, False)
    tmp_3 = tmp_2.permute([2, 0, 1])
    return tmp_3

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def embedding_permute_kernel(
    indices,
    weight,
    output,
    seq_len: tl.constexpr,
    embedding_dim: tl.constexpr,
    num_embeddings: tl.constexpr
):
    i = tl.program_id(0)
    j = tl.program_id(1)
    k = tl.arange(0, embedding_dim)
    
    idx = tl.load(indices + i * seq_len + j)
    weight_ptr = weight + idx * embedding_dim
    emb_vec = tl.load(weight_ptr + k)
    
    output_ptr = output + k * (seq_len * seq_len) + i * seq_len + j
    tl.store(output_ptr, emb_vec)

@torch.fx.wrap
def embedding_permute(in_0, in_1):
    seq_len = in_1.shape[0]
    embedding_dim = in_0.shape[1]
    output = torch.empty((embedding_dim, seq_len, seq_len),
                        device=in_1.device,
                        dtype=in_0.dtype)
    grid = (seq_len, seq_len)
    embedding_permute_kernel[grid](
        in_1,
        in_0,
        output,
        seq_len,
        embedding_dim,
        in_0.shape[0]
    )
    return output

def replacement_func():
    return embedding_permute