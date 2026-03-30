import torch
import triton
import triton.language as tl

def pattern(input_hidden, weight, scaling_factor):
    """
    Pattern: Linear transformation followed by element-wise multiplication
    Used in transformer models like Gemma and SmolLM3
    
    This matches the computation:
    linear_out = torch.nn.functional.linear(input_hidden, weight, None)
    result = scaling_factor * linear_out
    """
    linear_out = torch.nn.functional.linear(input_hidden, weight, None)
    result = scaling_factor * linear_out
    return result

def replacement_args(input_hidden, weight, scaling_factor):
    """Extract arguments for the fused kernel"""
    return (input_hidden, weight, scaling_factor)

@triton.jit
def fused_linear_multiply_kernel(
    hidden_ptr,           # Pointer to input hidden states [B, S, H]
    weight_ptr,           # Pointer to weight matrix [O, H] 
    scaling_ptr,          # Pointer to scaling factor [B, S, O]
    output_ptr,           # Pointer to output [B, S, O]
    batch_size,           # Batch size
    seq_len,              # Sequence length  
    hidden_dim,           # Hidden dimension
    output_dim,           # Output dimension
    # Strides
    hidden_stride_b,      # Stride for batch dim in hidden
    hidden_stride_s,      # Stride for seq dim in hidden
    hidden_stride_h,      # Stride for hidden dim in hidden
    weight_stride_o,      # Stride for output dim in weight
    weight_stride_h,      # Stride for hidden dim in weight
    scaling_stride_b,     # Stride for batch dim in scaling
    scaling_stride_s,     # Stride for seq dim in scaling
    scaling_stride_o,     # Stride for output dim in scaling
    output_stride_b,      # Stride for batch dim in output
    output_stride_s,      # Stride for seq dim in output
    output_stride_o,      # Stride for output dim in output
    BLOCK_SIZE_B: tl.constexpr,  # Block size for batch dimension
    BLOCK_SIZE_S: tl.constexpr,  # Block size for sequence dimension
    BLOCK_SIZE_H: tl.constexpr,  # Block size for hidden dimension
    BLOCK_SIZE_O: tl.constexpr,  # Block size for output dimension
):
    """Fused kernel: Linear transformation followed by element-wise multiplication"""
    
    # Get program indices for blocking
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_o = tl.program_id(2)
    
    # Compute ranges within each dimension
    batch_start = pid_b * BLOCK_SIZE_B
    seq_start = pid_s * BLOCK_SIZE_S
    output_start = pid_o * BLOCK_SIZE_O
    
    batch_end = min(batch_start + BLOCK_SIZE_B, batch_size)
    seq_end = min(seq_start + BLOCK_SIZE_S, seq_len)
    output_end = min(output_start + BLOCK_SIZE_O, output_dim)
    
    # Loop over hidden dimension for blocking
    for hid_offset in range(0, hidden_dim, BLOCK_SIZE_H):
        hid_end = min(hid_offset + BLOCK_SIZE_H, hidden_dim)
        
        # Initialize accumulator for linear part
        acc = tl.zeros((output_end - output_start,), dtype=tl.float32)
        
        # Compute linear transformation: sum over hidden dimension
        for hid_idx in range(hid_offset, hid_end):
            # Load weight for current hidden dimension
            weight_val = tl.load(
                weight_ptr + output_start * weight_stride_o + hid_idx * weight_stride_h,
                mask=(output_start < output_end),
                other=0.0
            ).to(tl.float32)
            
            # Load hidden states for current batch, seq, hidden
            hidden_val = tl.load(
                hidden_ptr + batch_start * hidden_stride_b + seq_start * hidden_stride_s + hid_idx * hidden_stride_h,
                mask=(batch_start < batch_size) and (seq_start < seq_len),
                other=0.0
            ).to(tl.float32)
            
            # Accumulate: weight * hidden
            acc += weight_val * hidden_val
        
        # Load scaling factor
        scaling_val = tl.load(
            scaling_ptr + batch_start * scaling_stride_b + seq_start * scaling_stride_s + output_start * scaling_stride_o,
            mask=(batch_start < batch_size) and (seq_start < seq_len) and (output_start < output_end),
            other=0.0
        ).to(tl.float32)
        
        # Store result: scaling * (linear_result)
        result = scaling_val * acc
        
        tl.store(
            output_ptr + batch_start * output_stride_b + seq_start * output_stride_s + output_start * output_stride_o,
            result.to(tl.float16 if tl.load(weight_ptr).dtype == tl.float16 else tl.bfloat16),
            mask=(batch_start < batch_size) and (seq_start < seq_len) and (output_start < output_end)
        )

@torch.fx.wrap
def fused_linear_multiply(input_hidden, weight, scaling_factor):
    """
    Fused implementation of linear transformation followed by element-wise multiplication
    Optimized using Triton for GPU performance
    """
    # Get tensor shapes and strides
    batch_size, seq_len, hidden_dim = input_hidden.shape
    output_dim = weight.shape[0]
    
    # Create output tensor
    output = torch.empty((batch_size, seq_len, output_dim), 
                        dtype=input_hidden.dtype, 
                        device=input_hidden.device)
    
    # Calculate strides
    hidden_stride_b = input_hidden.stride(0)
    hidden_stride_s = input_hidden.stride(1) 
    hidden_stride_h = input_hidden.stride(2)
    
    weight_stride_o = weight.stride(0)
    weight_stride_h = weight.stride(1)
    
    scaling_stride_b = scaling_factor.stride(0)
    scaling_stride_s = scaling_factor.stride(1)
    scaling_stride_o = scaling_factor.stride(2)
    
    output_stride_b = output.stride(0)
    output_stride_s = output.stride(1)
    output_stride_o = output.stride(2)
    
    # Block sizes for optimal GPU utilization
    BLOCK_SIZE_B = 1  # Process one batch at a time
    BLOCK_SIZE_S = 64  # Block sequence dimension
    BLOCK_SIZE_H = 32  # Block hidden dimension 
    BLOCK_SIZE_O = 64  # Block output dimension
    
    # Calculate grid dimensions
    grid_b = (batch_size + BLOCK_SIZE_B - 1) // BLOCK_SIZE_B
    grid_s = (seq_len + BLOCK_SIZE_S - 1) // BLOCK_SIZE_S
    grid_o = (output_dim + BLOCK_SIZE_O - 1) // BLOCK_SIZE_O
    
    # Launch kernel
    fused_linear_multiply_kernel[(
        grid_b, 
        grid_s, 
        grid_o
    )](
        hidden_ptr=input_hidden,
        weight_ptr=weight,
        scaling_ptr=scaling_factor,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        hidden_stride_b=hidden_stride_b,
        hidden_stride_s=hidden_stride_s,
        hidden_stride_h=hidden_stride_h,
        weight_stride_o=weight_stride_o,
        weight_stride_h=weight_stride_h,
        scaling_stride_b=scaling_stride_b,
        scaling_stride_s=scaling_stride_s,
        scaling_stride_o=scaling_stride_o,
        output_stride_b=output_stride_b,
        output_stride_s=output_stride_s,
        output_stride_o=output_stride_o,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_O=BLOCK_SIZE_O,
    )
    
    return output

def replacement_func():
    return fused_linear_multiply