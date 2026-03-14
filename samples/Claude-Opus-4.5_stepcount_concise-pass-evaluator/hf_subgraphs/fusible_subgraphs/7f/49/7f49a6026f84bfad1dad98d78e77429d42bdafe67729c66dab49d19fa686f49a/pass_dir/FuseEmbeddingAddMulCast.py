import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Pattern to match:
    - in_0: attention_mask [B, S]
    - in_1: input_ids [B, S]
    - in_2: position_embeddings_weight [P, D]
    - in_3: word_embeddings_weight [V, D]
    - in_4: position_ids [B, S]
    """
    tmp_4 = torch.nn.functional.embedding(in_1, in_3, 1, None, 2.0, False, False)
    tmp_5 = torch.nn.functional.embedding(in_4, in_2, 1, None, 2.0, False, False)
    tmp_6 = tmp_4 + tmp_5
    tmp_7 = in_0.unsqueeze(-1)
    tmp_8 = tmp_6 * tmp_7
    tmp_9 = tmp_8.to(torch.float32)
    return (tmp_9,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit 
def fused_embedding_simple(
    mask_ptr, ids_ptr, pos_emb_ptr, word_emb_ptr, posids_ptr, out_ptr,
    N, D: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return
    
    offs = tl.arange(0, D)
    
    wid = tl.load(ids_ptr + pid)
    pid_val = tl.load(posids_ptr + pid)
    m = tl.load(mask_ptr + pid).to(tl.float32)
    
    we = tl.load(word_emb_ptr + wid * D + offs)
    pe = tl.load(pos_emb_ptr + pid_val * D + offs)
    
    r = (we + pe) * m
    tl.store(out_ptr + pid * D + offs, r)


@torch.fx.wrap
def fused_embedding_add_mul_cast(attention_mask, input_ids, pos_emb_weight, word_emb_weight, position_ids):
    B, S = input_ids.shape
    D = word_emb_weight.shape[1]
    N = B * S
    
    out = torch.empty((B, S, D), dtype=torch.float32, device=input_ids.device)
    
    fused_embedding_simple[(N,)](
        attention_mask.view(-1),
        input_ids.view(-1),
        pos_emb_weight,
        word_emb_weight,
        position_ids.view(-1),
        out,
        N, D,
        num_warps=1,
    )
    
    return out


def replacement_func():
    return fused_embedding_add_mul_cast