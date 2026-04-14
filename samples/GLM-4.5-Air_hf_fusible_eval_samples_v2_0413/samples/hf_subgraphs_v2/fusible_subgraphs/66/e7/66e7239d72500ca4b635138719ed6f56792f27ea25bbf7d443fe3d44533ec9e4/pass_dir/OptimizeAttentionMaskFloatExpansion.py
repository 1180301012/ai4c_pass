import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact sequence from model.py
def pattern(in_0, in_1, in_2, in_3):
    # Match the computation from the model:
    tmp_4 = torch.nn.functional.layer_norm(in_3, (768,), in_2, in_1, 1e-12)
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.expand_as(tmp_4)
    tmp_7 = tmp_6.float()
    tmp_8 = tmp_4 * tmp_7
    return (tmp_7, tmp_8, tmp_4)  # Must return same structure as original model

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Triton kernel for optimized attention mask float expansion
@triton.jit
def optimized_attention_mask_kernel(
    attention_mask_ptr,
    layer_norm_output_ptr,
    expanded_mask_ptr,
    layer_norm_output_ptr_for_final_mul,  # Copy of layer_norm for final multiplication
    final_output_ptr,
    n_tokens,  # 1 * 16 = 16
    hidden_size,  # 768
    n_elements,  # n_tokens * hidden_size = 16 * 768 = 12288
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load attention mask value (attention_mask is [1, 16] -> we need the first row)
    # Since we're expanding, all positions in the same token row get the same value
    token_id = offsets // hidden_size
    attention_mask_value = tl.load(attention_mask_ptr + token_id, mask=token_id < n_tokens, other=0.0)
    
    # Load layer norm output element
    layer_norm_value = tl.load(layer_norm_output_ptr + offsets, mask=mask, other=0.0)
    
    # Create expanded float32 mask by casting attention mask value
    expanded_mask_float = tl.cast(attention_mask_value, tl.float32)
    
    # Store the expanded mask at the corresponding position
    tl.store(expanded_mask_ptr + offsets, expanded_mask_float, mask=mask)
    
    # Store layer norm output for final multiplication (needed for return)
    tl.store(layer_norm_output_ptr_for_final_mul + offsets, layer_norm_value, mask=mask)
    
    # Perform the multiplication: layer_norm * expanded_mask
    final_result = layer_norm_value * expanded_mask_float
    tl.store(final_output_ptr + offsets, final_result, mask=mask)

# Kernel wrapper - must be decorated with @torch.fx.wrap
@torch.fx.wrap
def optimized_attention_mask_forward(attention_mask, bias, weight, input_tensor):
    # Shapes: attention_mask [1, 16], bias [768], weight [768], input_tensor [1, 16, 768]
    layer_norm_output = torch.nn.functional.layer_norm(input_tensor, (768,), weight, bias, 1e-12)
    
    n_tokens = attention_mask.shape[1]  # 16
    hidden_size = input_tensor.shape[2]  # 768
    n_elements = n_tokens * hidden_size  # 12288
    
    # Choose optimal block size
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    expanded_mask = torch.empty((1, n_tokens, hidden_size), dtype=torch.float32, device=attention_mask.device)
    final_output = torch.empty_like(layer_norm_output, dtype=torch.float32)  # result will be float32
    layer_norm_copy = torch.empty_like(layer_norm_output)
    
    # Launch Triton kernel
    optimized_attention_mask_kernel[(num_programs,)](
        attention_mask_ptr=attention_mask,
        layer_norm_output_ptr=layer_norm_output,
        expanded_mask_ptr=expanded_mask,
        layer_norm_output_ptr_for_final_mul=layer_norm_copy,
        final_output_ptr=final_output,
        n_tokens=n_tokens,
        hidden_size=hidden_size,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return expanded_mask, final_output, layer_norm_copy

# Replacement function - returns the optimized function
def replacement_func():
    return optimized_attention_mask_forward