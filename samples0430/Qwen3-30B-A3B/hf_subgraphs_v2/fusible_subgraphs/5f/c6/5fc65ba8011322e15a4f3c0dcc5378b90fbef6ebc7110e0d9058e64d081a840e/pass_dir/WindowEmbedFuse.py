import torch
import triton
import triton.language as tl

@triton.jit
def window_embed_kernel(
    input_seq,
    weight,
    output,
    batch_size,
    seq_len,
    embedding_dim,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_EMB: tl.constexpr,
):
    block_idx_seq = tl.program_id(0)
    block_idx_emb = tl.program_id(1)
    batch_idx = tl.program_id(2)
    
    seq_start = block_idx_seq * BLOCK_SIZE_SEQ
    emb_start = block_idx_emb * BLOCK_SIZE_EMB
    
    thread_x = tl.thread_idx(0)
    thread_y = tl.thread_idx(1)
    
    i = seq_start + thread_x
    d = emb_start + thread_y
    
    if i >= seq_len or d >= 3 * embedding_dim:
        return
    
    if d < embedding_dim:
        idx = i + 1
    elif d < 2 * embedding_dim:
        idx = i
    else:
        idx = i - 1
    
    if idx < 0 or idx >= seq_len:
        tl.store(output + batch_idx * seq_len * (3 * embedding_dim) + i * (3 * embedding_dim) + d, 0.0)
    else:
        token_index = tl.load(input_seq + batch_idx * seq_len + idx)
        d_mod = d % embedding_dim
        val = tl.load(weight + token_index * embedding_dim + d_mod)
        tl.store(output + batch_idx * seq_len * (3 * embedding_dim) + i * (3 * embedding_dim) + d, val)

BLOCK_SIZE_SEQ = 32
BLOCK_SIZE_EMB = 64

@torch.fx.wrap
def window_embed(in_0, in_1):
    batch_size, seq_len = in_0.shape
    embedding_dim = in_1.shape[1]
    output = torch.empty(batch_size, seq_len, 3 * embedding_dim, dtype=in_1.dtype)
    grid = (triton.cdiv(seq_len, BLOCK_SIZE_SEQ), triton.cdiv(3 * embedding_dim, BLOCK_SIZE_EMB), batch_size)
    window_embed_kernel[grid](
        in_0,
        in_1,
        output,
        batch_size,
        seq_len,
        embedding_dim,
        BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
        BLOCK_SIZE_EMB=BLOCK_SIZE_EMB,
    )
    return output

def pattern(in_0, in_1):
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    tmp_3 = tmp_2[:, 1:]
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
    tmp_5 = tmp_2[:, :-1]
    tmp_6 = torch.nn.functional.pad(tmp_5, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
    tmp_7 = torch.cat([tmp_4, tmp_2, tmp_6], dim=2)
    return tmp_7

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    return window_embed