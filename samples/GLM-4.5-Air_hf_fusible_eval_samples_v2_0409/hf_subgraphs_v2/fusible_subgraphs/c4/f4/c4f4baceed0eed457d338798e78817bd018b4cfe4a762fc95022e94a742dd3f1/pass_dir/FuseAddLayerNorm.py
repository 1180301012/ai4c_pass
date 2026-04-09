import torch
import triton
import triton.language as tl

def pattern(tmp_7, in_2, in_1, in_0):
    tmp_8 = in_2 + tmp_7
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (384,), in_1, in_0, 1e-05)
    return tmp_8, tmp_9

def replacement_args(tmp_7, in_2, in_1, in_0):
    # Get the normalized feature dimension from the layer norm parameters
    normalized_shape = (384,) if in_0.shape[0] == 384 else (192,) if in_0.shape[0] == 192 else (96,)
    return (tmp_7, in_2, in_1, in_0, normalized_shape)

@triton.jit
def fused_add_layernorm_kernel(
    addend1_ptr,
    addend2_ptr,
    bias_ptr,
    weight_ptr,
    output_ptr,
    n_elements,
    normalized_shape,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.cdiv(n_elements, BLOCK_SIZE)
    
    # Each program handles BLOCK_SIZE consecutive elements
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    addend1 = tl.load(addend1_ptr + offsets, mask=mask, other=0.0)
    addend2 = tl.load(addend2_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    
    # Addition
    sum_result = addend1 + addend2
    
    # Layer normalization computation
    # Calculate mean
    local_sum = tl.sum(sum_result * mask.to(tl.float32))
    local_count = tl.sum(mask).to(tl.float32)
    mean = local_sum / local_count
    
    # Calculate variance
    diff = sum_result - mean
    local_var = tl.sum((diff * diff) * mask.to(tl.float32))
    var = local_var / local_count + eps
    
    # Normalize and scale
    inv_std = 1.0 / tl.sqrt(var)
    normalized = (sum_result - mean) * inv_std
    
    # Apply affine transformation
    output = weight * normalized + bias
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def optimized_fused_add_layernorm(addend1, addend2, bias, weight, normalized_shape):
    # Get input dimensions
    input_shape = addend1.shape
    n_elements = addend1.numel()
    
    # Create output tensor
    output_shape = input_shape  # Layer norm preserves shape
    output = torch.empty(output_shape, dtype=addend1.dtype, device=addend1.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid_size = (tl.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_add_layernorm_kernel[grid_size](
        addend1,
        addend2,
        bias,
        weight,
        output,
        n_elements,
        normalized_shape,
        1e-05,  # epsilon
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_fused_add_layernorm