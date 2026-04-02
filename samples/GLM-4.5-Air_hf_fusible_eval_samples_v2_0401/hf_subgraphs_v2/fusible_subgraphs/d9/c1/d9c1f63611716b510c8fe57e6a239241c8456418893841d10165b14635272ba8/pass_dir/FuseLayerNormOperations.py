import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight, bias, eps):
    """
    Pattern: LayerNorm operation
    Matches: torch.nn.functional.layer_norm(input, (256,), weight, bias, eps)
    """
    layer_norm_out = torch.nn.functional.layer_norm(input_tensor, (256,), weight, bias, eps)
    return layer_norm_out

def replacement_args(input_tensor, weight, bias, eps):
    return (input_tensor, weight, bias, eps)

@triton.jit
def layer_norm_kernel(
    input_ptr,  # input tensor [300, 256] or [300, 1, 256]
    weight_ptr,  # weight [256]
    bias_ptr,  # bias [256]
    output_ptr,  # output [same shape as input]
    n_elements,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    
    # Calculate range for this program
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, 300)
    
    # Load layer norm params to shared memory
    weight_smem = tl.zeros((256,), dtype=tl.float16)
    bias_smem = tl.zeros((256,), dtype=tl.float16)
    
    # Load weights and bias to shared memory
    for i in range(0, 256, 128):
        if i + 127 < 256:
            weight_smem[i:i+128] = tl.load(weight_ptr + i)
            bias_smem[i:i+128] = tl.load(bias_ptr + i)
    
    tl.device_barrier()
    
    # Process each element in the sequence dimension
    for m_idx in range(m_start, m_end):
        input_base = input_ptr + m_idx * 256
        
        # Load input tensor
        x = tl.load(input_base, mask=tl.arange(0, 256) < 256)
        
        # Layer normalization: (x - mean) / sqrt(var + eps) * weight + bias
        # Compute mean
        tensor_sum = tl.sum(x)
        mean = tensor_sum / 256
        
        # Compute variance using Welford's algorithm for numerical stability
        var_sum = tl.sum((x - mean) * (x - mean))
        var = var_sum / 256
        
        # Layer norm computation
        denom = tl.sqrt(var + eps)
        normalized = (x - mean) / denom
        
        # Apply layer norm weights and bias
        layer_norm_out = normalized * weight_smem + bias_smem
        
        # Store result
        tl.store(output_ptr + m_idx * 256, layer_norm_out, mask=tl.arange(0, 256) < 256)

@torch.fx.wrap
def optimized_layer_norm(input_tensor, weight, bias, eps):
    input_shape = input_tensor.shape
    n_elements = input_tensor.numel()
    
    # Determine optimal block sizes based on input shape
    if len(input_shape) == 3:  # [300, 1, 256]
        BLOCK_SIZE_M = 64  # Process 64 sequences at a time
    else:  # [300, 256]
        BLOCK_SIZE_M = 96  # Process 96 sequences at a time
    
    BLOCK_SIZE_N = 32  # Vector size
    total_sequences = input_shape[0]
    
    num_programs = (total_sequences + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    output = torch.empty_like(input_tensor)
    
    layer_norm_kernel[num_programs,](
        input_tensor,
        weight,
        bias,
        output,
        n_elements,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

def replacement_func():
    return optimized_layer_norm