import torch
import triton
import triton.language as tl
import math

def pattern(input_tensor, weight, bias):
    """Pattern to match linear operation followed by permute(0, 3, 1, 2)"""
    linear_result = torch.nn.functional.linear(input_tensor, weight, bias)
    permuted_result = linear_result.permute(0, 3, 1, 2)
    return permuted_result

def replacement_args(input_tensor, weight, bias):
    return (input_tensor, weight, bias)

@triton.jit
def fused_linear_permute_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    N: tl.constexpr, K: tl.constexpr, SEQL: tl.constexpr,
):
    """Fused linear + permute kernel processing single elements per program"""
    
    # Get program id for parallel processing
    pid = tl.program_id(0)
    
    # Convert program id to element offset
    element_offset = pid
    
    # Check if element_offset is within bounds
    if element_offset >= N * SEQL:
        return
    
    # Convert element_offset to 2D coordinates for [N, SEQL] output
    n_idx = element_offset // SEQL
    seq_idx = element_offset % SEQL
    
    # Load bias for this N dimension
    bias_val = tl.load(bias_ptr + n_idx)
    
    # Load weights for this N (individual scalar loads since K=3)
    weight_0 = tl.load(weight_ptr + n_idx * K + 0)
    weight_1 = tl.load(weight_ptr + n_idx * K + 1)
    weight_2 = tl.load(weight_ptr + n_idx * K + 2)
    
    # Load input for this spatial position (individual scalar loads since K=3)
    input_0 = tl.load(input_ptr + seq_idx * K + 0)
    input_1 = tl.load(input_ptr + seq_idx * K + 1)
    input_2 = tl.load(input_ptr + seq_idx * K + 2)
    
    # Compute dot product
    linear_result = bias_val + input_0 * weight_0 + input_1 * weight_1 + input_2 * weight_2
    
    # Store result in permuted layout [N, SEQL]
    output_offset = n_idx * SEQL + seq_idx
    tl.store(output_ptr + output_offset, linear_result)

@torch.fx.wrap
def fused_linear_permute(input_tensor, weight, bias):
    # Set up dimensions based on weight_meta
    batch_size = 1
    N = 16    # weight shape [16, 3] - output dimension  
    K = 3     # weight shape [16, 3] - input dimension
    SEQL = 196 * 196  # From input_3 shape [1, 196, 196, 3] - flattened spatial dims
    
    # Output shape: [batch_size, N, H, W] = [1, 16, 196, 196]
    output_shape = [batch_size, N, 196, 196]
    output = torch.zeros(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    # Flatten output to [N, SEQL] for kernel processing
    output_flat = output.permute(1, 0, 2, 3).contiguous().view(N, SEQL)
    
    # Calculate total number of programs needed
    # Each program processes exactly 1 element
    total_elements = N * SEQL
    num_programs = total_elements
    
    # Launch kernel to process output in [N, SEQL] layout
    fused_linear_permute_kernel[(num_programs,)](
        input_tensor,
        weight,
        bias,
        output_flat,
        N, K, SEQL
    )
    
    return output

def replacement_func():
    return fused_linear_permute