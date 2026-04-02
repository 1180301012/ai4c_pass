import torch
import triton
import triton.language as tl

def sigmoid_input_a, input_b, output_c, eps):
    # Layer norm operation
    layer_norm_a = torch.nn.functional.layer_norm(input_a, (256,), weight_a, bias_a, eps)
    layer_norm_b = torch.nn.functional.layer_norm(input_b, (256,), weight_b, bias_b, eps)
    
    # First path: sigmoid -> unsqueeze -> multiplication
    sigmoid_a = sigmoid_input_a.sigmoid()
    unsqueeze_b = layer_norm_b.unsqueeze(-2)
    multiply1 = sigmoid_a * unsqueeze_b
    
    # Second path: sigmoid -> multiplication
    sigmoid_b = layer_norm_b.sigmoid()
    multiply2 = sigmoid_b * layer_norm_b
    
    # Addition
    result = multiply1 + multiply2
    
    return multiply1, multiply2, result, layer_norm_b

def replacement_argsigmoid_input_a, layer_norm_a, weight_a, bias_a, eps, layer_norm_b, weight_b, bias_b):
    return (sigmoid_input_a, layer_norm_a, weight_a, bias_b, ep
    weight_a, bias_a, eps, layer_norm_b, weight_):
    """
    Pattern: sigmoid -> unsqueeze -> multiply + sigmoid -> multiply -> add
    For the sequence: tmp_11 * tmp_14 + tmp_10 * tmp_13
    """
    sigmoid_input_a, input_b, output_c, eps):
    # Compute laye
        layer_norm_a = torch.nn.functional.layer_norm(input_a, (256,), weight_a, bias_a, eps)
        layer_norm_b = torch.nn.functional.layer_norm(input_b, (256,), weight_b, bias_b, eps)
        
        # First path: sigmoid -> unsqueeze -> multiplication
        sigmoid_tmp_a = torch.nn.functional.layer_norm(input_a, (256,), weight_a, bias_b, eps)
        sigmoid_b = layer_norm_b.sigmoid()
        multiply1 = sigmoid_a * torch.sigmoid(sigmoid_input_a)
        unsqueeze_b = layer_norm_b.unsqueeze(-2)
        multiply1 = sigmoid_a * unsqueeze_b
        
        # Second path: sigmoid for input gate on layer_norm_b
        sigmoid_b = layer_norm_b.sigmoid()
        multiply2 = sigmoid_b * layer_norm_b
        
        # Addition and then another sigmoid
        add_result = multiply1 + multiply2
        final_sigmoid = add_result.sigmoid()
        
        return add_result, final_sigmoid

def replacement_args(sigmoid_input_a, layer_norm_a, weight_a, bias_a, eps, layer_norm_b, weight_b, bias_b):
    return (sigmoid_input_a, layer_norm_a, weight_a, bias_a, eps, layer_norm_b, weight_b, bias_b)

@triton.jit
def fused_elementwise_kernel(
    sigmoid_input_a_ptr,  # input for sigmoid [300, 1, 256]
    layer_norm_a_ptr,  # first layer norm output [300, 1, 256]
    weight_a_ptr,  # layer norm weights [256] (shared memory)
    bias_a_ptr,  # layer norm bias [256] (shared memory)
    layer_norm_b_ptr,  # second layer norm output [300, 256] 
    weight_b_ptr,  # layer norm weights [256] (shared memory)
    bias_b_ptr,  # layer norm bias [256] (shared memory)
    add_result_ptr,  # final add result [300, 1, 256]
    final_ptr,  # final output after sigmoid [300, 1, 256]
    n_elements,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    
    # Calculate range for this program
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, 300)
    
    # Load layer norm params to shared memory
    weight_a_smem = tl.zeros((256,), dtype=tl.float16)
    bias_a_smem = tl.zeros((256,), dtype=tl.float16)
    weight_b_smem = tl.zeros((256,), dtype=tl.float16)
    bias_b_smem = tl.zeros((256,), dtype=tl.float16)
    
    # Load weights and bias to shared memory
    for i in range(0, 256, 128):
        if i + 127 < 256:
            weight_a_smem[i:i+128] = tl.load(weight_a_ptr + i)
            bias_a_smem[i:i+128] = tl.load(bias_a_ptr + i)
            weight_b_smem[i:i+128] = tl.load(weight_b_ptr + i)
            bias_b_smem[i:i+128] = tl.load(bias_b_ptr + i)
    
    tl.device_barrier()
    
    # Process each element
    for m_idx in range(m_start, m_end):
        # Load input tensors
        sigmoid_input_a = tl.load(sigmoid_input_a_ptr + m_idx * 256, mask=tl.arange(0, 256) < 256)
        layer_norm_a = tl.load(layer_norm_a_ptr + m_idx * 256, mask=tl.arange(0, 256) < 256)
        layer_norm_b = tl.load(layer_norm_b_ptr + m_idx * 256, mask=tl.arange(0, 256) < 256)
        
        # Compute sigmoid operations
        sigmoid_a = 1.0 / (1.0 + tl.exp(-sigmoid_input_a))
        sigmoid_b = 1.0 / (1.0 + tl.exp(-layer_norm_b))
        
        # First multiplication: sigmoid_a * unsqueeze(layer_norm_b)
        # unsqueeze(-2) means add dimension at position -2, so [300, 256] -> [300, 1, 256]
        # This creates a broadcast pattern where layer_norm_b works for all positions
        broadcast_mult1 = sigmoid_a * sigmoid_b  # [300, 1, 256] * broadcast [300, 1, 256]
        
        # Second multiplication: sigmoid_b * layer_norm_a
        broadcast_mult2 = sigmoid_b * layer_norm_a  # [300, 1, 256] * [300, 1, 256]
        
        # Addition
        add_result = broadcast_mult1 + broadcast_mult2
        
        # Final sigmoid
        final_result = 1.0 / (1.0 + tl.exp(-add_result))
        
        # Store results
        tl.store(add_result_ptr + m_idx * 256, add_result, mask=tl.arange(0, 256) < 256)
        tl.store(final_ptr + m_idx * 256, final_result, mask=tl.arange(0, 256) < 256)

@torch.fx.wrap
def fused_elementwise_ops(sigmoid_input_a, layer_norm_a, weight_a, bias_a, eps, layer_norm_b, weight_b, bias_b):
    output_shape = sigmoid_input_a.shape
    n_elements = sigmoid_input_a.numel()
    
    # Determine block sizes
    BLOCK_SIZE_M = 64  # Process 64 elements at a time
    BLOCK_SIZE_N = 32  # Vector size
    total_elements = output_shape[0]
    
    num_programs = (total_elements + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    add_result = torch.empty_like(sigmoid_input_a)
    final_result = torch.empty_like(sigmoid_input_a)
    
    fused_elementwise_kernel[num_programs,](
        sigmoid_input_a,
        layer_norm_a,
        weight_a,
        bias_a,
        layer_norm_b,
        weight_b,
        bias_b,
        add_result,
        final_result,
        n_elements,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return add_result, final_result

def replacement_func():
    return fused_elementwise_ops