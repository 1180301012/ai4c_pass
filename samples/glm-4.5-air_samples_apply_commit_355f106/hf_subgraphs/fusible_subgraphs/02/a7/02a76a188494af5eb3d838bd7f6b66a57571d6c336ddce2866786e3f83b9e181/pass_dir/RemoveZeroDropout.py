import torch
import triton
import triton.language as tl

def pattern(input_ids, embedding_weight, norm_weight):
    """
    Pattern: embedding -> layer_norm -> dropout (with p=0.0)
    The dropout with p=0.0 is identity, so we can remove it entirely.
    """
    # This matches the sequence: embedding -> layer_norm -> identity_dropout
    # Since dropout with p=0.0 is identity, we can return the layer_norm output directly
    embedding_out = torch.nn.functional.embedding(input_ids, embedding_weight, 50283, None, 2.0, False, False)
    layer_norm_out = torch.nn.functional.layer_norm(embedding_out, (768,), norm_weight, None, 1e-05)
    # Dropout with p=0.0 is identity, so we return layer_norm output
    return layer_norm_out

def replacement_args(input_ids, embedding_weight, norm_weight):
    """
    Extract arguments for the replacement.
    Since we're removing dropout completely, we only need the inputs to the computation.
    """
    return (input_ids, embedding_weight, norm_weight)

@torch.fx.wrap
def optimized_embedding_layer_norm(input_ids, embedding_weight, norm_weight, norm_bias=None, eps=1e-05):
    """
    Optimized fused embedding + layer norm without dropout
    """
    # Get input shapes
    seq_len = input_ids.shape[-1]
    embedding_dim = embedding_weight.shape[-1]
    vocab_size = embedding_weight.shape[0]
    
    # Number of elements in batch
    n_elements = input_ids.numel()
    
    # Optimize block size based on typical embedding sizes
    BLOCK_SIZE = 128
    
    # Output tensor
    output = torch.empty_like(input_ids, dtype=torch.float32)
    
    @triton.jit
    def embedding_kernel(
        input_ids_ptr,
        embedding_weight_ptr,
        output_ptr,
        vocab_size,
        embedding_dim,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Each program handles a contiguous block
        block_start = tl.program_id(0) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load input IDs (flatten for easier processing)
        input_ids = tl.load(input_ids_ptr + offsets, mask=mask, other=0)
        
        # Convert to embedding indices (handling potential padding)
        valid_mask = input_ids < vocab_size
        input_ids = tl.where(valid_mask, input_ids, 0)
        
        # Initialize output
        output = tl.zeros((BLOCK_SIZE, embedding_dim), dtype=tl.float32)
        
        # Process each embedding in the block
        for k in range(0, BLOCK_SIZE, 1):
            if offsets[k] < n_elements:
                idx = input_ids[k]
                if idx < vocab_size:
                    # Load embedding vector
                    emb_ptr = embedding_weight_ptr + idx * embedding_dim
                    embedding = tl.load(emb_ptr + tl.arange(0, embedding_dim), mask=tl.arange(0, embedding_dim) < embedding_dim)
                    output[k] = embedding
        
        # Store results
        tl.store(output_ptr + offsets * embedding_dim, output.reshape(-1), mask=mask)
    
    # Launch kernel
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    embedding_kernel[(num_programs,)](
        input_ids,
        embedding_weight,
        output,
        vocab_size,
        embedding_dim,
        n_elements,
        BLOCK_SIZE,
    )
    
    # Apply layer normalization
    if norm_bias is not None:
        output = output * norm_weight + norm_bias
    else:
        output = output * norm_weight
    
    # Compute mean and variance for layer norm
    mean = output.mean(dim=-1, keepdim=True)
    var = output.var(dim=-1, keepdim=True, unbiased=False)
    output = (output - mean) / torch.sqrt(var.to(torch.float32) + eps)
    
    return output

def replacement_func():
    """
    Return the optimized function that removes dropout
    """
    return optimized_embedding_layer_norm