import torch
from torch import device
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_9 = in_4.unsqueeze(0)
    tmp_10 = tmp_9 + 2
    tmp_11 = torch.nn.functional.embedding(tmp_10, in_1, None, None, 2.0, False, False)
    tmp_12 = tmp_11.to(device(type='cuda', index=0))
    tmp_13 = in_0 + tmp_12
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (tmp_13.shape[-1],), in_3, in_2, 1e-05)
    tmp_15 = torch.nn.functional.dropout(tmp_14, p = 0.1, training = False)
    return tmp_15


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def optimized_embedding_kernel(
    input_ids_ptr,
    embedding_table_ptr,
    output_ptr,
    n_tokens,
    n_embed,
    BLOCK_SIZE_TOKENS: tl.constexpr,
    BLOCK_SIZE_EMBED: tl.constexpr,
):
    # Each program processes one token
    token_id = tl.program_id(0)
    
    if token_id >= n_tokens:
        return
    
    # Load input ID for this token
    input_id = tl.load(input_ids_ptr + token_id)
    
    # Embed offset calculation: input_id * 2.0 (from original computation)
    embed_offset = (input_id.to(tl.int32) * 2).to(tl.int64)
    
    # Load embedding vector for this token
    embed_ptr = embedding_table_ptr + embed_offset * n_embed
    
    # Initialize output with zeros
    output_vec = tl.zeros((BLOCK_SIZE_EMBED,), dtype=tl.float32)
    
    # Load embedding vector
    for k in range(0, n_embed, BLOCK_SIZE_EMBED):
        mask = k + tl.arange(0, BLOCK_SIZE_EMBED) < n_embed
        embed_val = tl.load(embed_ptr + k + tl.arange(0, BLOCK_SIZE_EMBED), mask=mask, other=0.0)
        output_vec = output_vec + embed_val
    
    # Store the embedding result
    tl.store(output_ptr + token_id * n_embed + tl.arange(0, BLOCK_SIZE_EMBED), 
             output_vec, mask=(tl.arange(0, BLOCK_SIZE_EMBED) < n_embed))


@triton.jit
def optimized_layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    hidden_size,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input, weight, and bias
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets % hidden_size, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets % hidden_size, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x) / tl.sum(mask.to(tl.float32))
    
    # Compute variance
    x_centered = x - mean
    variance = tl.sum(x_centered * x_centered) / tl.sum(mask.to(tl.float32))
    
    # Layer norm
    norm_x = (x_centered) / tl.sqrt(variance + eps)
    output = norm_x * weight + bias
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit  
def optimized_dropout_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    p: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply dropout (scale by 1/(1-p) during training)
    scale = 1.0 / (1.0 - p) if p > 0 else 1.0
    output = x * scale
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


@torch.fx.wrap
def optimized_embedding_computation_chain(in_0, in_1, in_2, in_3, in_4):
    # Step 1: Process input positions (unsqueeze + add 2)
    positions = in_4.unsqueeze(0) + 2
    
    # Get dimensions
    n_tokens = positions.shape[1]
    n_embed = in_1.shape[1]
    
    # Embedding lookup
    embeddings = torch.empty(1, n_tokens, n_embed, dtype=torch.float32, device=in_1.device)
    
    # Launch embedding kernel
    BLOCK_SIZE_EMBED = 128
    optimized_embedding_kernel[(n_tokens,)](
        input_ids_ptr=positions,
        embedding_table_ptr=in_1,
        output_ptr=embeddings,
        n_tokens=n_tokens,
        n_embed=n_embed,
        BLOCK_SIZE_TOKENS=1,
        BLOCK_SIZE_EMBED=BLOCK_SIZE_EMBED,
    )
    
    # Reshape embeddings to match expected shape
    embeddings = embeddings.squeeze(0)  # Remove the unsqueeze dimension
    
    # Add to input embeddings
    combined = in_0 + embeddings
    
    # Layer normalization
    ln_output = torch.empty_like(combined)
    total_elements = combined.numel()
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_layer_norm_kernel[(num_programs,)](
        input_ptr=combined,
        weight_ptr=in_3,
        bias_ptr=in_2,
        output_ptr=ln_output,
        n_elements=total_elements,
        hidden_size=in_3.shape[0],
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Dropout
    dropout_output = torch.empty_like(ln_output)
    optimized_dropout_kernel[(num_programs,)](
        input_ptr=ln_output,
        output_ptr=dropout_output,
        n_elements=total_elements,
        p=0.1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return dropout_output


def replacement_func():
    return optimized_embedding_computation_chain