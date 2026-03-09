import torch
import triton
import triton.language as tl

def pattern(input_ids, embedding_weight, norm_weight):
    """
    Pattern: embedding lookup + layer_norm + identity_dropout
    Focus on optimizing just the embedding lookup portion
    """
    # Match the embedding operation specifically
    embedding_out = torch.nn.functional.embedding(input_ids, embedding_weight, 50283, None, 2.0, False, False)
    # We include layer_norm in the pattern to match graph structure
    layer_norm_out = torch.nn.functional.layer_norm(embedding_out, (768,), norm_weight, None, 1e-05)
    # Dropout with p=0.0 is identity
    return layer_norm_out

def replacement_args(input_ids, embedding_weight, norm_weight):
    """
    Extract arguments for embedding optimization
    """
    return (input_ids, embedding_weight, norm_weight)

@torch.fx.wrap
def optimized_embedding_lookup(input_ids, embedding_weight, norm_weight, norm_bias=None, eps=1e-05):
    """
    Highly optimized embedding lookup with autotuning
    """
    # Get shapes
    batch_size, seq_len = input_ids.shape
    embedding_dim = embedding_weight.shape[-1]
    vocab_size = embedding_weight.shape[0]
    
    n_elements = input_ids.numel()
    
    # Output tensor
    output = torch.empty((batch_size, seq_len, embedding_dim), dtype=torch.float32, device=input_ids.device)
    
    # Autotune configuration for different input sizes
    @triton.autotune(
        configs=[
            triton.Config(num_warps=4, num_stages=2),  # Small inputs
            triton.Config(num_warps=8, num_stages=3),  # Medium inputs  
            triton.Config(num_warps=16, num_stages=4), # Large inputs
        ],
        key=['n_elements', 'embedding_dim'],
        prune_configs_by={
            'early_stopping': 5,
            'metrics': ['time'],
            'perf_model': None,
            'top_k': 1,
        },
    )
    @triton.jit
    def optimized_embedding_kernel(
        input_ids_ptr,
        embedding_weight_ptr,
        output_ptr,
        vocab_size,
        embedding_dim,
        batch_size,
        seq_len,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        num_warps: tl.constexpr,
        num_stages: tl.constexpr,
    ):
        # Each program handles a block of elements
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load input IDs with vectorization
        input_ids = tl.load(input_ids_ptr + offsets, mask=mask, other=0)
        
        # Initialize output accumulator
        embeddings = tl.zeros((BLOCK_SIZE, embedding_dim), dtype=tl.float32)
        
        # Process embeddings with better memory access pattern
        for k in range(tl.cdiv(BLOCK_SIZE, 4)):
            idx = k * 4
            if idx + 3 < BLOCK_SIZE and offsets[idx] + 3 < n_elements:
                # Vectorized load for 4 consecutive indices
                idx0, idx1, idx2, idx3 = input_ids[idx], input_ids[idx+1], input_ids[idx+2], input_ids[idx+3]
                
                # Process 4 embeddings together
                for i, idx_val in enumerate([idx0, idx1, idx2, idx3]):
                    if idx_val < vocab_size:
                        emb_ptr = embedding_weight_ptr + idx_val * embedding_dim
                        emb_vec = tl.load(emb_ptr + tl.arange(0, embedding_dim), 
                                        mask=tl.arange(0, embedding_dim) < embedding_dim)
                        embeddings[idx + i] = emb_vec
            elif idx < BLOCK_SIZE and offsets[idx] < n_elements:
                # Single element fallback
                idx_val = input_ids[idx]
                if idx_val < vocab_size:
                    emb_ptr = embedding_weight_ptr + idx_val * embedding_dim
                    emb_vec = tl.load(emb_ptr + tl.arange(0, embedding_dim), 
                                    mask=tl.arange(0, embedding_dim) < embedding_dim)
                    embeddings[idx] = emb_vec
        
        # Store results with coalesced memory access
        output_flat = embeddings.reshape(-1)
        tl.store(output_ptr + offsets * embedding_dim, output_flat, mask=mask)
    
    # Different block sizes for different input sizes
    if n_elements < 1024:
        BLOCK_SIZE = 64
    elif n_elements < 8192:
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 256
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch with autotuning
    optimized_embedding_kernel[(num_programs,)](
        input_ids,
        embedding_weight,
        output,
        vocab_size,
        embedding_dim,
        batch_size,
        seq_len,
        n_elements,
        BLOCK_SIZE,
        num_warps=8,  # Default, will be autotuned
        num_stages=3, # Default, will be autotuned
    )
    
    # Apply remaining layer norm operations separately (could be another optimization)
    if norm_bias is not None:
        output = output * norm_weight + norm_bias
    else:
        output = output * norm_weight
    
    # Layer norm
    mean = output.mean(dim=-1, keepdim=True)
    var = output.var(dim=-1, keepdim=True, unbiased=False)
    output = (output - mean) / torch.sqrt(var.to(torch.float32) + eps)
    
    return output

def replacement_func():
    """
    Return optimized embedding lookup function
    """
    return optimized_embedding_lookup