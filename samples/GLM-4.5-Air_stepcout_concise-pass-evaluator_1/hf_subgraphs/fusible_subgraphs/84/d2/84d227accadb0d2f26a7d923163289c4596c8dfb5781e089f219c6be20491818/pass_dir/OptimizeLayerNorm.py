import torch
import triton
import triton.language as tl

def pattern(tmp_6, tmp_2, tmp_1, eps):
    # Match the exact layer_norm call structure from the model
    normalized_shape = (tmp_6.shape[-1],)
    return torch.nn.functional.layer_norm(tmp_6, normalized_shape, tmp_2, tmp_1, eps)

def replacement_args(tmp_6, tmp_2, tmp_1, eps):
    return (tmp_6, tmp_2, tmp_1, eps)

@triton.jit
def layernorm_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, seq_len, hidden_size,
    eps, BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element
    batch_idx = tl.program_id(0)
    
    # Calculate offsets for this batch element
    elem_offset = tl.arange(0, BLOCK_SIZE)
    col_mask = elem_offset < hidden_size
    
    # Calculate base address for this batch element
    batch_base = batch_idx * seq_len * hidden_size
    
    for seq_idx in range(seq_len):
        # Calculate base address for current sequence position
        seq_base = batch_base + seq_idx * hidden_size
        
        # Load input for this position [hidden_size]
        x = tl.load(
            input_ptr + seq_base + elem_offset,
            mask=col_mask,
            other=0.0
        ).to(tl.float32)
        
        # Load weights and bias [hidden_size]
        weight = tl.load(
            weight_ptr + elem_offset,
            mask=col_mask,
            other=1.0
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + elem_offset,
            mask=col_mask,
            other=0.0
        ).to(tl.float32)
        
        # Compute mean
        mean = tl.sum(x) / hidden_size
        
        # Compute variance
        x_centered = x - mean
        variance = tl.sum(x_centered * x_centered) / hidden_size
        
        # Compute std dev
        std_dev = tl.sqrt(variance + eps)
        
        # LayerNorm: (x - mean) / std_dev * weight + bias
        normalized = x_centered / std_dev
        result = normalized * weight + bias
        
        # Store result
        tl.store(
            output_ptr + seq_base + elem_offset,
            result,
            mask=col_mask
        )

@torch.fx.wrap
def optimized_layernorm(input_tensor, weight, bias, eps):
    batch_size, seq_len, hidden_size = input_tensor.shape
    
    BLOCK_SIZE = 256
    grid = (batch_size * seq_len,)
    
    output = torch.empty_like(input_tensor, dtype=torch.float32)
    
    # Use half precision if supported for better performance
    if input_tensor.dtype == torch.float16:
        input_tensor_fp = input_tensor.to(torch.float32)
        weight_fp = weight.to(torch.float32)
        bias_fp = bias.to(torch.float32)
        eps_fp = float(eps)
    else:
        input_tensor_fp = input_tensor
        weight_fp = weight
        bias_fp = bias
        eps_fp = float(eps)
    
    layernorm_kernel[grid](
        input_tensor_fp, weight_fp, bias_fp, output,
        batch_size, seq_len, hidden_size,
        eps_fp, BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_layernorm