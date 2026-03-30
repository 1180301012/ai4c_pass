import torch
import triton
import triton.language as tl

def pattern(tmp_word_embeddings, tmp_pos_embeddings):
    """
    Pattern matching for embedding addition:
    - tmp_word_embeddings = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    - tmp_pos_embeddings = torch.nn.functional.embedding(tmp_14, in_3, 1, None, 2.0, False, False)  
    - tmp_16 = tmp_word_embeddings + tmp_pos_embeddings
    Pattern matches the addition of two embedding tensors that were computed separately.
    """
    tmp_16 = tmp_word_embeddings + tmp_pos_embeddings
    return tmp_16

def replacement_args(tmp_10, tmp_15):
    return (tmp_10, tmp_15)

@triton.jit
def fused_embedding_kernel(
    word_emb_ptr,
    pos_emb_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.cdiv(batch_size * seq_len * hidden_dim, BLOCK_SIZE)
    
    if pid >= num_programs:
        return
        
    # Compute which element this program handles
    total_elements = batch_size * seq_len * hidden_dim
    element_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = element_idx < total_elements
    
    # Load embeddings directly from input tensors and add them
    word_emb = tl.load(word_emb_ptr + element_idx, mask=mask, other=0.0)
    pos_emb = tl.load(pos_emb_ptr + element_idx, mask=mask, other=0.0)
    
    # Add embeddings and store
    output = word_emb + pos_emb
    tl.store(output_ptr + element_idx, output, mask=mask)

@torch.fx.wrap
def fused_embedding_forward(tmp_word_embeddings, tmp_pos_embeddings):
    batch_size, seq_len, hidden_dim = tmp_word_embeddings.shape
    
    # Check that inputs match expected shape
    assert tmp_word_embeddings.shape == tmp_pos_embeddings.shape
    assert tmp_word_embeddings.dtype == tmp_pos_embeddings.dtype
    
    # Use the existing embedding tensors, no need to recompute indices
    output = torch.empty_like(tmp_word_embeddings)
    
    # Use medium block size for optimal GPU occupancy and memory access pattern
    BLOCK_SIZE = 512   # Balance between cache locality and GPU utilization
    
    # Calculate grid size
    total_elements = batch_size * seq_len * hidden_dim
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_programs,)
    
    fused_embedding_kernel[grid](
        word_emb_ptr=tmp_word_embeddings,
        pos_emb_ptr=tmp_pos_embeddings,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_embedding_forward