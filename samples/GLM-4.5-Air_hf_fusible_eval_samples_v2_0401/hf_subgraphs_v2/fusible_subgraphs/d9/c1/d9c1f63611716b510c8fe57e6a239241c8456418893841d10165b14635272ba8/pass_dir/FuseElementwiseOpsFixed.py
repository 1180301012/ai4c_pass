import torch
import triton
import triton.language as tl

def pattern(sigmoid_input_a, sigmoid_input_b, layer_norm_b_input, layer_norm_weight_b, layer_norm_bias_b, eps):
    """
    Pattern: sigmoid_input_a * unsqueeze(layer_norm_b) + sigmoid_input_b * layer_norm_b
    For the sequence: tmp_11 * tmp_14 + tmp_10 * tmp_13
    """
    # Compute layer norm on the input
    layer_norm_b = torch.nn.functional.layer_norm(layer_norm_b_input, (256,), layer_norm_weight_b, layer_norm_bias_b, eps)
    
    # First multiplication: sigmoid_input_a * unsqueeze(layer_norm_b)
    unsqueeze_layer_norm_b = layer_norm_b.unsqueeze(-2)  # [300, 256] -> [300, 1, 256]
    multiply1 = sigmoid_input_a * unsqueeze_layer_norm_b
    
    # Second multiplication: sigmoid_input_b * layer_norm_b (broadcasted)
    broadcast_layer_norm_b = layer_norm_b.unsqueeze(-2)  # [300, 256] -> [300, 1, 256]
    multiply2 = sigmoid_input_b * broadcast_layer_norm_b
    
    # Addition
    result = multiply1 + multiply2
    
    return multiply1, multiply2, result

def replacement_args(sigmoid_input_a, sigmoid_input_b, layer_norm_b_input, layer_norm_weight_b, layer_norm_bias_b, eps):
    return (sigmoid_input_a, sigmoid_input_b, layer_norm_b_input, layer_norm_weight_b, layer_norm_bias_b, eps)

@triton.jit
def fused_elementwise_kernel(
    sigmoid_input_a_ptr,  # [300, 1, 256]
    sigmoid_input_b_ptr,  # [300, 1, 256]
    layer_norm_b_input_ptr,  # [300, 256]
    layer_norm_weight_b_ptr,  # [256]
    layer_norm_bias_b_ptr,  # [256]
    multiply1_ptr,  # [300, 1, 256]
    multiply2_ptr,  # [300, 1, 256]
    result_ptr,  # [300, 1, 256]
    n_elements,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    
    # Calculate range for this program
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, 300)
    
    # Load layer norm params to shared memory
    weight_b_smem = tl.zeros((256,), dtype=tl.float16)
    bias_b_smem = tl.zeros((256,), dtype=tl.float16)
    
    # Load weights and bias to shared memory
    for i in range(0, 256, 128):
        if i + 127 < 256:
            weight_b_smem[i:i+128] = tl.load(layer_norm_weight_b_ptr + i)
            bias_b_smem[i:i+128] = tl.load(layer_norm_bias_b_ptr + i)
    
    tl.device_barrier()
    
    # Process each element
    for m_idx in range(m_start, m_end):
        # Load input tensors
        sigmoid_a = tl.load(sigmoid_input_a_ptr + m_idx * 256, mask=tl.arange(0, 256) < 256)
        sigmoid_b = tl.load(sigmoid_input_b_ptr + m_idx * 256, mask=tl.arange(0, 256) < 256)
        layer_norm_b_input = tl.load(layer_norm_b_input_ptr + m_idx * 256, mask=tl.arange(0, 256) < 256)
        
        # Layer normalization: (x - mean) / sqrt(var + eps) * weight + bias
        # Compute mean
        tensor_sum = tl.sum(layer_norm_b_input)
        mean = tensor_sum / 256
        
        # Compute variance
        var_sum = tl.sum((layer_norm_b_input - mean) * (layer_norm_b_input - mean))
        var = var_sum / 256
        
        # Layer norm computation
        denom = tl.sqrt(var + eps)
        layer_norm_b = (layer_norm_b_input - mean) / denom
        
        # Apply layer norm weights and bias
        layer_norm_b = layer_norm_b * weight_b_smem + bias_b_smem
        
        # Create broadcasted version by replicating across the sequence dimension
        # unsqueeze(-2) operation: [300, 256] -> [300, 1, 256]
        # We simulate this by treating it as available for all positions
        layer_norm_b_broadcasted = layer_norm_b
        
        # First multiplication: sigmoid_a * unsqueeze(layer_norm_b)
        multiply1 = sigmoid_a * layer_norm_b_broadcasted
        
        # Second multiplication: sigmoid_b * layer_norm_b (broadcasted)
        multiply2 = sigmoid_b * layer_norm_b_broadcasted
        
        # Addition
        result = multiply1 + multiply2
        
        # Store results
        tl.store(multiply1_ptr + m_idx * 256, multiply1, mask=tl.arange(0, 256) < 256)
        tl.store(multiply2_ptr + m_idx * 256, multiply2, mask=tl.arange(0, 256) < 256)
        tl.store(result_ptr + m_idx * 256, result, mask=tl.arange(0, 256) < 256)

@torch.fx.wrap
def fused_elementwise_ops(sigmoid_input_a, sigmoid_input_b, layer_norm_b_input, layer_norm_weight_b, layer_norm_bias_b, eps):
    output_shape = sigmoid_input_a.shape
    n_elements = sigmoid_input_a.numel()
    
    # Determine block sizes
    BLOCK_SIZE_M = 96  # Process 96 sequences at a time
    total_sequences = output_shape[0]
    
    num_programs = (total_sequences + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    multiply1 = torch.empty_like(sigmoid_input_a)
    multiply2 = torch.empty_like(sigmoid_input_a)
    result = torch.empty_like(sigmoid_input_a)
    
    fused_elementwise_kernel[num_programs,](
        sigmoid_input_a,
        sigmoid_input_b,
        layer_norm_b_input,
        layer_norm_weight_b,
        layer_norm_bias_b,
        multiply1,
        multiply2,
        result,
        n_elements,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    
    return multiply1, multiply2, result

def replacement_func():
    return fused_elementwise_ops