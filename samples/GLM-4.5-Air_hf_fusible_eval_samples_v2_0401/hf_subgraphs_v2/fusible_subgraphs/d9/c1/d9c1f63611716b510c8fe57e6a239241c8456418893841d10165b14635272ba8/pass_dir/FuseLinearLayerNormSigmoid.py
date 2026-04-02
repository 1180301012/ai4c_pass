import torch
import triton
import triton.language as tl

def pattern(linear_input, linear_weight, linear_bias, ln_weight, ln_bias, eps):
    """
    Pattern: Linear + LayerNorm + Sigmoid 
    Matches: linear -> layer_norm -> sigmoid
    """
    linear_out = torch.nn.functional.linear(linear_input, linear_weight, linear_bias)
    layer_norm_out = torch.nn.functional.layer_norm(linear_out, (256,), ln_weight, ln_bias, eps)
    sigmoid_out = layer_norm_out.sigmoid()
    return sigmoid_out, linear_out  # Return both to maintain observability

def replacement_args(linear_input, linear_weight, linear_bias, ln_weight, ln_bias, eps):
    return (linear_input, linear_weight, linear_bias, ln_weight, ln_bias, eps)

@triton.jit
def fused_linear_layer_norm_sigmoid_kernel(
    x_ptr,  # linear_input [300, 1, 256]
    weight_ptr,  # linear_weight [256, 256]
    bias_ptr,  # linear_bias [256]
    ln_weight_ptr,  # ln_weight [256]
    ln_bias_ptr,  # ln_bias [256]
    out_ptr,  # output [300, 1, 256]
    linear_out_ptr,  # intermediate linear output [300, 1, 256]
    n_elements,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    
    # Calculate range for this program
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, 300)
    
    # Allocate shared memory for small tensor
    ln_weight_smem = tl.zeros((256,), dtype=tl.float16)
    ln_bias_smem = tl.zeros((256,), dtype=tl.float16)
    
    # Load layer norm params to shared memory
    for i in range(0, 256, 128):
        if i + 127 < 256:
            ln_weight_smem[i:i+128] = tl.load(ln_weight_ptr + i)
            ln_bias_smem[i:i+128] = tl.load(ln_bias_ptr + i)
    
    tl.device_barrier()
    
    # Process each element in the sequence dimension
    for m_idx in range(m_start, m_end):
        input_base = x_ptr + m_idx * 256
        
        # Compute linear operation manually for better fusion
        accum = tl.zeros((256,), dtype=tl.float16)
        
        # Matmul: [1, 256] x [256, 256] -> [256]
        for n in range(0, 256, 32):
            for k in range(0, 256, 32):
                # Load input chunk
                x_chunk = tl.load(input_base + k, mask=(k + tl.arange(0, 32)) < 256)
                
                # Load weight chunk
                weight_chunk = tl.load(weight_ptr + n * 256 + k, 
                                       mask=(k + tl.arange(0, 32)) < 256)
                
                # Accumulate results
                accum[n:n+32] += x_chunk * weight_chunk
        
        # Add bias
        accum += tl.load(bias_ptr, mask=tl.arange(0, 256) < 256)
        
        # Store linear output
        tl.store(linear_out_ptr + m_idx * 256, accum, mask=tl.arange(0, 256) < 256)
        
        # Layer normalization: (x - mean) / sqrt(var + eps) * weight + bias
        # Compute mean
        tensor_sum = tl.sum(accum)
        mean = tensor_sum / 256
        
        # Compute variance
        var_sum = tl.sum((accum - mean) * (accum - mean))
        var = var_sum / 256
        
        # Layer norm computation
        denom = tl.sqrt(var + eps)
        normalized = (accum - mean) / denom
        
        # Apply layer norm weights and bias
        layer_norm_out = normalized * ln_weight_smem + ln_bias_smem
        
        # Apply sigmoid
        sigmoid_out = 1.0 / (1.0 + tl.exp(-layer_norm_out))
        
        # Store final output
        tl.store(out_ptr + m_idx * 256, sigmoid_out, mask=tl.arange(0, 256) < 256)

@torch.fx.wrap
def fused_linear_layer_norm_sigmoid(linear_input, linear_weight, linear_bias, ln_weight, ln_bias, eps):
    linear_input_shape = linear_input.shape
    n_elements = linear_input.numel()
    
    # Determine optimal block sizes
    BLOCK_SIZE_M = 32  # Process 32 sequences at a time
    BLOCK_SIZE_N = 32  # Vector size
    total_sequences = linear_input_shape[0]
    
    num_programs_m = (total_sequences + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    linear_out = torch.empty_like(linear_input)
    sigmoid_out = torch.empty_like(linear_input)
    
    fused_linear_layer_norm_sigmoid_kernel[num_programs_m,](
        linear_input,
        linear_weight,
        linear_bias,
        ln_weight,
        ln_bias,
        sigmoid_out,
        linear_out,
        n_elements,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return sigmoid_out, linear_out

def replacement_func():
    return fused_linear_layer_norm_sigmoid