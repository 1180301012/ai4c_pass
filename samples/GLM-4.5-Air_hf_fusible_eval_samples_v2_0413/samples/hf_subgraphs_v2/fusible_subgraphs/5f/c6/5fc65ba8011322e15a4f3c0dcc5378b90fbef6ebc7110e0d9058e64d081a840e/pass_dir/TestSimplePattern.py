import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def simple_embedding_kernel(
    input_ids_ptr,
    embedding_weight_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    embed_dim: tl.constexpr,
    BLOCK_EMBED: tl.constexpr,
):
    # Simple but efficient program id approach
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    embed_idx = tl.program_id(2)
    
    # Calculate output offset for this program
    output_offset = (batch_idx * seq_len + seq_idx) * embed_dim + embed_idx * BLOCK_EMBED
    
    # Efficient memory access - load input id once
    input_idx = batch_idx * seq_len + seq_idx
    input_id = tl.load(input_ids_ptr + input_idx)
    
    # Calculate embedding base address
    embed_base = input_id * embed_dim
    
    # Coalesced memory access for embedding data
    read_offset = embed_base + embed_idx * BLOCK_EMBED
    embed_data = tl.load(
        embedding_weight_ptr + read_offset + tl.arange(0, BLOCK_EMBED),
        mask=(embed_idx * BLOCK_EMBED + tl.arange(0, BLOCK_EMBED)) < embed_dim,
        other=0.0
    )
    
    # Coalesced store to output
    tl.store(
        output_ptr + output_offset + tl.arange(0, BLOCK_EMBED),
        embed_data,
        mask=(embed_idx * BLOCK_EMBED + tl.arange(0, BLOCK_EMBED)) < embed_dim
    )

@torch.fx.wrap
def simple_embedding(in_0, in_1):
    batch_size = in_0.shape[0]
    seq_len = in_0.shape[1]
    embed_dim = in_1.shape[1]
    
    output = torch.zeros((batch_size, seq_len, embed_dim), dtype=in_1.dtype, device=in_1.device)
    
    # Optimized block sizes for better performance
    BLOCK_EMBED = 128  # Larger block size for better memory utilization
    BLOCK_BATCH = 1    # One batch element per program (simpler)
    BLOCK_SEQ = 1      # One sequence position per program (simpler)
    
    num_batches = (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH
    num_seqs = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    num_embeds = (embed_dim + BLOCK_EMBED - 1) // BLOCK_EMBED
    
    # Pass original tensors - the kernel will handle indexing internally
    simple_embedding_kernel[(num_batches, num_seqs, num_embeds)](
        in_0,
        in_1,
        output,
        batch_size,
        seq_len,
        embed_dim,
        BLOCK_EMBED,
    )
    
    return output

def replacement_func():
    return simple_embedding