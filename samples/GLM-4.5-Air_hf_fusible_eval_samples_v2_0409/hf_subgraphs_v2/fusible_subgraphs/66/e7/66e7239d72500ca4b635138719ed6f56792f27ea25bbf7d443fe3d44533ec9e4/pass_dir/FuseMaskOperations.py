import torch
import triton
import triton.language as tl

# Pattern matching function for the complete expand + convert sequence
def pattern(attention_mask, layer_norm_output):
    """
    Match the sequence: unsqueeze(-1) -> expand_as -> float()
    This fuses three operations into one efficient kernel
    Args:
        attention_mask: [1, 16], torch.int64
        layer_norm_output: [1, 16, 768], torch.float16/bfloat16 (used for shape reference)
    Returns:
        expanded_float_mask: [1, 16, 768], torch.float32
    """
    tmp_5 = attention_mask.unsqueeze(-1)
    tmp_6 = tmp_5.expand_as(layer_norm_output)
    tmp_7 = tmp_6.float()
    return tmp_7

def replacement_args(attention_mask, layer_norm_output):
    return (attention_mask, layer_norm_output)

@triton.jit
def fused_mask_kernel(
    attention_mask_ptr,
    layer_norm_output_ptr,
    output_ptr,
    batch, seq_len, hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that fuses expand and convert operations
    Converts attention mask from [batch, seq_len] to [batch, seq_len, hidden_size]
    and converts dtype from int64 to float32 in one efficient kernel
    """
    pid = tl.program_id(0)
    
    # Calculate offsets for this program
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < batch * seq_len * hidden_size
    
    # Load and expand attention mask efficiently
    # For each (batch, seq) position, we need to broadcast to all hidden dimensions
    flat_idx = offset
    batch_idx = flat_idx // (seq_len * hidden_size)
    seq_idx = (flat_idx % (seq_len * hidden_size)) // hidden_size
    hidden_idx = flat_idx % hidden_size
    
    # Load attention mask value for this batch and sequence
    attention_value = tl.load(attention_mask_ptr + batch_idx * seq_len + seq_idx)
    
    # Convert to float32 and broadcast (Triton handles type casting)
    float_attention = attention_value
    
    # Store the expanded mask
    tl.store(output_ptr + offset, float_attention, mask=mask)

@torch.fx.wrap
def fused_mask_operations(attention_mask, layer_norm_output):
    """
    Fused function that combines unsqueeze, expand_as, and float conversion
    Args:
        attention_mask: [1, 16], torch.int64
        layer_norm_output: [1, 16, 768], torch.float16/bfloat16
    Returns:
        expanded_float_mask: [1, 16, 768], torch.float32
    """
    batch, seq_len, hidden_size = layer_norm_output.shape
    total_elements = batch * seq_len * hidden_size
    
    # Optimal block size for better GPU utilization
    BLOCK_SIZE = 512  # Larger block size for better throughput
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output_shape = (batch, seq_len, hidden_size)
    output = torch.empty(output_shape, dtype=torch.float32, device=attention_mask.device)
    
    # Launch kernel
    fused_mask_kernel[(num_programs,)](
        attention_mask_ptr=attention_mask,
        layer_norm_output_ptr=layer_norm_output,
        output_ptr=output,
        batch=batch,
        seq_len=seq_len,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_mask_operations