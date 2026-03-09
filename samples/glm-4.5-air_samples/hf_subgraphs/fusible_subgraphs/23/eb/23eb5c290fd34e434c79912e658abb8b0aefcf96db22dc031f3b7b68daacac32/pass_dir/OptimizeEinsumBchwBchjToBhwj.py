import torch
import triton
import triton.language as tl

def pattern(key, query):
    # This is the einsum pattern we want to optimize
    attention_scores = torch.functional.einsum('bchw,bchj->bhwj', query, key)
    return attention_scores

@triton.jit
def optimized_einsum_kernel(
    query_ptr,    # [B, C, H, W]
    key_ptr,      # [B, C, H, J] 
    out_ptr,      # [B, H, W, J]
    B_offset: tl.constexpr,
):
    # Each program handles one spatial (h, w) position
    h = tl.program_id(0)
    w = tl.program_id(1)
    
    # Process all j positions for this (h, w)
    j_indices = tl.arange(0, 64)
    mask = j_indices < 64
    
    # Compute sum over C for each j position
    sum_vals = tl.zeros((64,), dtype=tl.float32)
    
    for j_idx in range(64):
        j = j_idx
        if j < 64:
            # Compute sum over C: query[b,c,h,w] * key[b,c,h,j]
            total = 0.0
            for c in range(64):
                # Load query[b,c,h,w]
                query_offset = B_offset * 64 * 64 * 64 + c * 64 * 64 + h * 64 + w
                query_val = tl.load(query_ptr + query_offset)
                
                # Load key[b,c,h,j]  
                key_offset = B_offset * 64 * 64 * 64 + c * 64 * 64 + h * 64 + j
                key_val = tl.load(key_ptr + key_offset)
                
                total += query_val * key_val
            
            sum_vals[j_idx] = total
    
    # Store all results for this (h, w) position
    out_offset = B_offset * 64 * 64 * 64 + h * 64 * 64 + w * 64 + j_indices
    tl.store(out_ptr + out_offset, sum_vals, mask=mask)

@torch.fx.wrap
def optimized_einsum(key, query):
    B, C, H, W = query.shape
    _, _, _, J = key.shape
    
    # Create output tensor [B, H, W, J]
    output = torch.empty((B, H, W, J), dtype=query.dtype, device=query.device)
    
    # For each batch, launch kernel with 2D grid over spatial dimensions
    for batch_idx in range(B):
        # Launch kernel with 2D grid: [H, W] = [64, 64]
        optimized_einsum_kernel[(64, 64)](
            query, key, output, batch_idx
        )
    
    return output

def replacement_args(key, query):
    return (key, query)

def replacement_func():
    return optimized_einsum