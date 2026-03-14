import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, layernorm_out):
    """Matches the final add-multiply pattern"""
    tmp_14 = in_1 * layernorm_out
    tmp_1 = layernorm_out = None
    tmp_15 = tmp_14 + in_0
    tmp_14 = None
    return tmp_15

def replacement_args(in_0, in_1, layernorm_out):
    return (in_0, in_1, layernorm_out)

@triton.jit
def addmul_kernel(
    bias_ptr,
    weight_ptr,
    normalized_ptr,
    out_ptr,
    bias_stride,
    weight_stride,
    normalized_batch_stride,
    normalized_seq_stride,
    normalized_hidden_stride,
    out_batch_stride,
    out_seq_stride,
    out_hidden_stride,
    batch_size,
    seq_len,
    hidden_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Get program IDs  
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # hidden dimension
    
    # Compute memory addresses
    bias_hidden_offset = pid_n * BLOCK_N
    weight_hidden_offset = pid_n * BLOCK_N
    
    normalized_batch_offset = pid_m * normalized_batch_stride
    normalized_hidden_offset = pid_n * BLOCK_N
    
    out_batch_offset = pid_m * out_batch_stride
    out_hidden_offset = pid_n * BLOCK_N
    
    # Process each sequence position
    for s in range(0, seq_len):
        # Load weight and bias for this hidden dimension
        weight = tl.load(weight_ptr + weight_hidden_offset + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < hidden_size, other=0.0)
        bias = tl.load(bias_ptr + bias_hidden_offset + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < hidden_size, other=0.0)
        
        # Load normalized tensor for this sequence position
        normalized_ptr_local = normalized_batch_offset + s * normalized_seq_stride + normalized_hidden_offset
        normalized = tl.load(normalized_ptr_local + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < hidden_size, other=0.0)
        
        # Apply weight: weight * normalized
        weighted = weight * normalized
        
        # Add bias: weighted + bias
        result = weighted + bias
        
        # Store result
        out_ptr_local = out_batch_offset + s * out_seq_stride + out_hidden_offset
        tl.store(out_ptr_local + tl.arange(0, BLOCK_N), result, mask=tl.arange(0, BLOCK_N) < hidden_size)

@torch.fx.wrap  
def fused_addmul(bias, weight, normalized):
    # Determine input shapes
    batch_size, seq_len, hidden_size = normalized.shape
    
    # Set optimal block sizes 
    BLOCK_N = min(256, hidden_size)
    BLOCK_M = 64  # Block size for sequence dimension
    
    # Calculate grid size
    grid_m = (batch_size + BLOCK_M - 1) // BLOCK_M
    grid_n = (hidden_size + BLOCK_N - 1) // BLOCK_N
    
    # Prepare output tensor
    out = torch.zeros_like(normalized)
    
    # Launch kernel - fix stride handling for 1D tensors (bias/weight)
    bias_s0 = bias.stride(0) if bias.dim() > 0 else 1
    bias_s1 = bias.stride(-1) if bias.dim() > 0 else 1
    weight_s0 = weight.stride(0) if weight.dim() > 0 else 1
    weight_s1 = weight.stride(-1) if weight.dim() > 0 else 1
    
    addmul_kernel[grid_m, grid_n](
        bias, weight, normalized, out,
        bias_s0, bias_s1,
        weight_s0, weight_s1,
        normalized.stride(0), normalized.stride(1), normalized.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        batch_size, seq_len, hidden_size,
        BLOCK_M, BLOCK_N
    )
    
    return out

def replacement_func():
    return fused_addmul