import torch
import triton
import triton.language as tl

def pattern(input_ids, embedding_weight, norm_weight):
    """
    Pattern: embedding -> layer_norm -> identity_dropout (p=0.0)
    Fuse embedding + layer_norm + dropout elimination into single operation
    """
    # This matches the full sequence: embedding -> layer_norm -> identity_dropout
    embedding_out = torch.nn.functional.embedding(input_ids, embedding_weight, 50283, None, 2.0, False, False)
    layer_norm_out = torch.nn.functional.layer_norm(embedding_out, (768,), norm_weight, None, 1e-05)
    # Dropout with p=0.0 is identity, so we return layer_norm output
    return layer_norm_out

def replacement_args(input_ids, embedding_weight, norm_weight):
    """
    Extract arguments for the fused replacement.
    """
    return (input_ids, embedding_weight, norm_weight)

@torch.fx.wrap
def fused_embedding_layer_norm(input_ids, embedding_weight, norm_weight, norm_bias=None, eps=1e-05):
    """
    Fused Triton kernel for embedding lookup + layer norm optimization
    """
    # Get input shapes
    n_elements = input_ids.numel()
    embedding_dim = embedding_weight.shape[-1]
    vocab_size = embedding_weight.shape[0]
    
    # Optimize based on typical embedding dimensions
    BLOCK_SIZE = 256
    
    # Output tensor
    output = torch.empty((n_elements, embedding_dim), dtype=torch.float32, device=input_ids.device)
    
    @triton.jit
    def fused_embedding_norm_kernel(
        input_ids_ptr,
        embedding_weight_ptr,
        norm_weight_ptr,
        output_ptr,
        vocab_size,
        embedding_dim,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Each program handles a contiguous block of input tokens
        block_start = tl.program_id(0) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load input IDs
        input_ids = tl.load(input_ids_ptr + offsets, mask=mask, other=0)
        
        # # Initialize output accumulator - use 1D arrays dynamically sized
        embeddings = tl.zeros(BLOCK_SIZE * embedding_dim, dtype=tl.float32)
        norms = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
        norms_sq = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
        
        # Process each position in the block
        for k in range(0, BLOCK_SIZE):
            if offsets[k] < n_elements:
                idx = input_ids[k]
                if idx < vocab_size:
                    # Load embedding with proper bounds checking
                    emb_ptr = embedding_weight_ptr + idx * embedding_dim
                    emb_vec = tl.load(emb_ptr + tl.arange(0, embedding_dim), 
                                     mask=tl.arange(0, embedding_dim) < embedding_dim)
                    # Store in flat array
                    emb_start = k * embedding_dim
                    for i in range(embedding_dim):
                        embeddings[emb_start + i] = emb_vec[i]
        
        # Apply layer normalization weights and compute statistics
        if norm_weight_ptr is not None:
            for k in range(0, BLOCK_SIZE):
                if offsets[k] < n_elements:
                    # Load norm weight
                    norm_weight = tl.load(norm_weight_ptr + tl.arange(0, embedding_dim), 
                                        mask=tl.arange(0, embedding_dim) < embedding_dim)
                    # Apply weight and compute statistics
                    emb_start = k * embedding_dim
                    weighted_emb_sum = 0.0
                    weighted_emb_sq_sum = 0.0
                    
                    for i in range(embedding_dim):
                        emb_val = embeddings[emb_start + i] * norm_weight[i]
                        embeddings[emb_start + i] = emb_val
                        weighted_emb_sum += emb_val
                        weighted_emb_sq_sum += emb_val * emb_val
                    
                    norms[k] = weighted_emb_sum
                    norms_sq[k] = weighted_emb_sq_sum
        
        # Compute mean and variance and apply normalization
        for k in range(0, BLOCK_SIZE):
            if offsets[k] < n_elements:
                mean = norms[k] / embedding_dim
                var = (norms_sq[k] / embedding_dim) - (mean * mean)
                std = tl.sqrt(var + eps)
                
                # Apply layer normalization
                emb_start = k * embedding_dim
                for i in range(embedding_dim):
                    embeddings[emb_start + i] = (embeddings[emb_start + i] - mean) / std
        
        # Store results with coalesced memory access
        tl.store(output_ptr + offsets * embedding_dim, embeddings, mask=mask)
    
    # Launch kernel
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    fused_embedding_norm_kernel[(num_programs,)](
        input_ids,
        embedding_weight,
        norm_weight,
        output,
        vocab_size,
        embedding_dim,
        n_elements,
        BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """
    Return the fused embedding + layer norm function
    """
    return fused_embedding_layer_norm