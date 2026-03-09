import torch
import triton
import triton.language as tl

def pattern(energy_H_1, key, query):
    # Match the exact computation from the model
    attention_scores = torch.functional.einsum('bchw,bchj->bhwj', query, key)
    concatenated = torch.cat([energy_H_1, attention_scores], dim=-1)
    softmax_output = torch.nn.functional.softmax(concatenated, dim=-1)
    sliced_output = softmax_output[..., slice(None, 64, None)]
    return softmax_output, sliced_output

@triton.jit
def optimized_attention_kernel(
    energy_ptr,    # [B, H, W, C]
    query_ptr,     # [B, C, H, W] -> need to transpose to [B, H, W, C]
    key_ptr,       # [B, C, H, J] -> need to transpose to [B, H, J, C]
    output_full_ptr,   # [B, H, W, 128] - full softmax output
    output_slice_ptr,  # [B, H, W, 64] - sliced output
    B, C, H, W, J,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial position (h, w) in the batch
    batch_id = tl.program_id(0)
    hw_id = tl.program_id(1)
    
    # Convert hw_id to h, w coordinates
    h_id = hw_id // W
    w_id = hw_id % W
    
    # Work on energy part (first C elements)
    energy_offset = batch_id * H * W * C + h_id * W * C + w_id * C + tl.arange(0, C)
    energy_vals = tl.load(energy_ptr + energy_offset, mask=tl.arange(0, C) < C, other=0.0)
    
    # Compute attention scores efficiently
    # Process query: [B, C, H, W] -> transpose to [B, H, W, C] for this position
    # For position (h, w), we want query[:, :, h, w] @ key.transpose(-1, -2)
    
    # Load query for this position: [C]
    query_offset = batch_id * C * H * W + tl.arange(0, C) * H * W + h_id * W + w_id
    query_vals = tl.load(query_ptr + query_offset, mask=tl.arange(0, C) < C, other=0.0)
    
    # Load and transpose key: [B, C, H, J] -> we need key[:, :, h, :] for this position
    # This becomes [J, C] for matrix multiplication
    key_vals = tl.zeros((J, C), dtype=tl.float32)
    
    # Load key slice for this h position and all w positions in J
    for j in range(0, J, BLOCK_SIZE):
        j_range = j + tl.arange(0, min(BLOCK_SIZE, J - j))
        j_mask = j_range < J
        
        key_offset = (
            batch_id * C * H * J + 
            j_range * C * H + 
            h_id * C
        )
        key_block = tl.load(key_ptr + key_offset, mask=j_mask[:, None] & tl.arange(0, C)[None, :] < C, other=0.0)
        
        # Transpose and store
        key_vals += tl.view(key_block, (len(j_range), C))
    
    # Compute dot product: query_vals @ key_vals.T  -> [J]
    attention_scores = tl.dot(query_vals, key_vals, trans_b=True)
    
    # Concatenate energy + attention scores
    combined = tl.concat([energy_vals, attention_scores], axis=0)
    
    # Apply softmax
    max_val = tl.max(combined)
    shifted = combined - max_val
    exp_sum = tl.sum(tl.exp(shifted))
    softmax_result = tl.exp(shifted) / exp_sum
    
    # Store full output
    full_offset = batch_id * H * W * 128 + hw_id * 128 + tl.arange(0, 128)
    tl.store(output_full_ptr + full_offset, softmax_result, mask=tl.arange(0, 128) < 128)
    
    # Store sliced output (first 64 elements)
    slice_offset = batch_id * H * W * 64 + hw_id * 64 + tl.arange(0, 64)
    tl.store(output_slice_ptr + slice_offset, softmax_result[:64], mask=tl.arange(0, 64) < 64)

@torch.fx.wrap  
def optimized_attention_kernel_wrapper(energy_H_1, key, query):
    B, C, H, W = energy_H_1.shape
    _, _, _, J = key.shape
    
    # Output tensors
    output_full = torch.empty((B, H, W, 128), dtype=energy_H_1.dtype, device=energy_H_1.device)
    output_slice = torch.empty((B, H, W, 64), dtype=energy_H_1.dtype, device=energy_H_1.device)
    
    # Launch kernel
    total_hw_elements = B * H * W
    optimized_attention_kernel[(total_hw_elements,)](
        energy_H_1, query, key,
        output_full, output_slice,
        B, C, H, W, J,
        block_size=1024
    )
    
    return output_full, output_slice

def replacement_args(energy_H_1, key, query):
    return (energy_H_1, key, query)

def replacement_func():
    return optimized_attention_kernel_wrapper