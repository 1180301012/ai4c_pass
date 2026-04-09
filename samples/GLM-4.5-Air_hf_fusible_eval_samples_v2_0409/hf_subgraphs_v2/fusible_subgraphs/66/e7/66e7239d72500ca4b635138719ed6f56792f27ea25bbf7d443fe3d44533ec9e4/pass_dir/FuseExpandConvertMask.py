import torch
import triton
import triton.language as tl

# Pattern matching function for expand + convert sequence
def pattern(attention_mask, layer_norm_output):
    """
    Match the sequence: unsqueeze(-1) -> expand_as -> float()
    Args:
        attention_mask: [1, 16], torch.int64
        layer_norm_output: [1, 16, 768], torch.float16/bfloat16
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
def expand_convert_kernel(
    attention_mask_ptr,
    output_ptr,
    layer_norm_batch,
    layer_norm_seq_len,
    layer_norm_hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that fuses expand and convert operations
    Converts attention mask from [batch, seq_len] to [batch, seq_len, hidden_size]
    and converts dtype from int64 to float32 in one kernel
    """
    # Each program handles one sequence position in the batch
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    hidden_pid = tl.program_id(2)
    
    # Hidden dimension is expanded, so we handle it with vectorized loads
    hidden_offset = hidden_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = hidden_offset < layer_norm_hidden_size
    
    # Load attention mask value (scalar for this batch and sequence position)
    attention_value = tl.load(attention_mask_ptr + batch_idx * layer_norm_seq_len + seq_idx)
    
    # Convert to float32 and broadcast to all hidden dimensions
    float_attention = tl.float32(attention_value)
    
    # Store expanded mask
    output_ptr = output_ptr + batch_idx * layer_norm_seq_len * layer_norm_hidden_size + seq_idx * layer_norm_hidden_size
    tl.store(output_ptr + hidden_offset, float_attention, mask=mask)

@torch.fx.wrap
def fused_expand_convert(attention_mask, layer_norm_output):
    """
    Wrapper function for the fused expand+convert kernel
    Args:
        attention_mask: [1, 16], torch.int64
        layer_norm_output: [1, 16, 768], torch.float16/bfloat16
    Returns:
        output: [1, 16, 768], torch.float32
    """
    batch, seq_len = attention_mask.shape
    _, _, hidden_size = layer_norm_output.shape
    
    # Determine optimal block size
    BLOCK_SIZE = 256
    num_programs = (hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output_shape = (batch, seq_len, hidden_size)
    output = torch.empty(output_shape, dtype=torch.float32, device=attention_mask.device)
    
    # Launch kernel
    expand_convert_kernel[(batch, seq_len, num_programs)](
        attention_mask_ptr=attention_mask,
        output_ptr=output,
        layer_norm_batch=batch,
        layer_norm_seq_len=seq_len,
        layer_norm_hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_expand_convert