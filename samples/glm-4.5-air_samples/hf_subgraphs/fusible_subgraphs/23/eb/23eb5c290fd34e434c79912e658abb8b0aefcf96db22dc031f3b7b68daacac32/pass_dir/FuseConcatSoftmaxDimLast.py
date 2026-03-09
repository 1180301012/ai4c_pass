import torch
import triton
import triton.language as tl

def pattern(energy_H_1, key, query):
    # Einsum operation - this is our base computation
    attention_scores = torch.functional.einsum('bchw,bchj->bhwj', query, key)
    
    # Concatenation operation
    concatenated = torch.cat([energy_H_1, attention_scores], dim=-1)
    
    # Softmax operation 
    softmax_output = torch.nn.functional.softmax(concatenated, dim=-1)
    
    # Slicing operation
    sliced_output = softmax_output[..., slice(None, 64, None)]
    
    return attention_scores, softmax_output, concatenated, sliced_output

@triton.jit
def fused_concat_softmax_kernel(
    energy_ptr,    # [B, H, W, C1]
    scores_ptr,    # [B, H, W, C2] 
    out_ptr,       # [B, H, W, C1+C2]
    n_hw,          # Number of H*W elements per batch
    c1,            # Size of first dimension (64)
    c2,            # Size of second dimension (64)
    block_size: tl.constexpr,
):
    # Each program handles one batch element's H*W elements
    batch_id = tl.program_id(0)
    hw_id = tl.program_id(1)
    
    # Calculate offset for this batch element
    batch_offset = batch_id * n_hw * (c1 + c2)
    hw_offset = hw_id * (c1 + c2)
    offset = batch_offset + hw_offset
    
    # Load energy_H_1 part (first c1 elements)
    energy_energy = tl.load(energy_ptr + batch_id * n_hw * c1 + hw_id * c1 + tl.arange(0, c1), mask=tl.arange(0, c1) < c1, other=0.0)
    
    # Load attention scores part (next c2 elements)
    scores_energy = tl.load(scores_ptr + batch_id * n_hw * c2 + hw_id * c2 + tl.arange(0, c2), mask=tl.arange(0, c2) < c2, other=0.0)
    
    # Concatenate: energy_H_1 + attention_scores
    combined = tl.concat([energy_energy, scores_energy], axis=0)
    
    # Apply softmax along the concatenated dimension
    max_val = tl.max(combined)
    shifted = combined - max_val
    exp_sum = tl.sum(tl.exp(shifted))
    softmax_result = tl.exp(shifted) / exp_sum
    
    # Store result
    tl.store(out_ptr + offset, softmax_result)

@torch.fx.wrap
def fused_concat_softmax(energy_H_1, attention_scores, original_shape):
    B, H, W, C = original_shape
    total_elements = B * H * W
    c1 = C  # energy_H_1 has size 64 on last dim
    c2 = C  # attention_scores has size 64 on last dim
    
    # Output will have size [B, H, W, C+C] = [B, H, W, 128]
    output_shape = (B, H, W, c1 + c2)
    output = torch.empty(output_shape, dtype=energy_H_1.dtype, device=energy_H_1.device)
    
    # Launch kernel
    fused_concat_softmax_kernel[(B, total_elements // (H * W))](
        energy_H_1, attention_scores, output,
        H * W, c1, c2,
        block_size=1024
    )
    
    return output

def replacement_args(energy_H_1, key, query):
    # We need the attention scores for the fused operation
    attention_scores = torch.functional.einsum('bchw,bchj->bhwj', query, key)
    original_shape = energy_H_1.shape
    return (energy_H_1, attention_scores, original_shape)

def replacement_func():
    return fused_concat_softmax