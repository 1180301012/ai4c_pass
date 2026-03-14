import torch
import triton
import triton.language as tl

def pattern(inputs_embeds, token_type_embeddings):
    # Simple fusion of inputs_embeds + token_type_embeddings
    return inputs_embeds + token_type_embeddings

def replacement_args(inputs_embeds, token_type_embeddings):
    return (inputs_embeds, token_type_embeddings)

@triton.jit
def fused_add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data with proper bounds checking
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Add element-wise
    result = x + y
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_embedding_add(inputs_embeds, token_type_embeddings):
    # Ensure tensors are on the same device
    if inputs_embeds.device != token_type_embeddings.device:
        token_type_embeddings = token_type_embeddings.to(inputs_embeds.device)
    
    # Ensure tensors have the same dtype
    if inputs_embeds.dtype != token_type_embeddings.dtype:
        token_type_embeddings = token_type_embeddings.to(inputs_embeds.dtype)
    
    total_elements = inputs_embeds.numel()
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(inputs_embeds)
    
    # Launch kernel
    fused_add_kernel[(num_programs,)](
        inputs_embeds, token_type_embeddings, output,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_embedding_add