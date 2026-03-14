import torch
import triton
import triton.language as tl


def pattern(layer_norm_weight, layer_norm_bias, input_tensor1, input_tensor2):
    # Addition operation
    added = input_tensor1 + input_tensor2
    # Layer normalization 
    result = torch.nn.functional.layer_norm(added, (512,), layer_norm_weight, layer_norm_bias, 1e-05)
    added = None
    return result


def replacement_args(layer_norm_weight, layer_norm_bias, input_tensor1, input_tensor2):
    return (layer_norm_weight, layer_norm_bias, input_tensor1, input_tensor2)


@triton.jit
def fused_layer_norm_addition_kernel(
    x_ptr,
    y_ptr,
    weight_ptr, 
    bias_ptr,
    out_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one warp
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Fusion: Addition
    added = x + y
    
    # Load layer norm parameters
    weight = tl.load(weight_ptr + pid % 512, mask=None)
    bias = tl.load(bias_ptr + pid % 512, mask=None)
    
    # Mean and variance calculation
    mean = tl.sum(added) / n_elements
    centered = added - mean
    variance = tl.sum(centered * centered) / n_elements
    std = tl.sqrt(variance + eps)
    
    # Layer normalization
    norm = (centered / std) * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, norm, mask=mask)


@torch.fx.wrap
def fused_layer_norm_addition(layer_norm_weight, layer_norm_bias, input_tensor1, input_tensor2):
    # Use Triton kernel for all cases
    if input_tensor1.dim() == 3:
        # Input shape: [batch, seq_len, hidden_size] - optimized case
        n_elements = input_tensor1.numel()
        BLOCK_SIZE = 512
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        out = torch.empty_like(input_tensor1)
        
        fused_layer_norm_addition_kernel[(num_programs,)](
            input_tensor1,
            input_tensor2,
            layer_norm_weight,
            layer_norm_bias,
            out,
            n_elements,
            1e-05,
            BLOCK_SIZE
        )
        
        return out
    else:
        # Use simple element-wise operations for different dimensions
        added = input_tensor1 + input_tensor2
        # Apply normalization manually without torch.nn.functional
        mean = added.mean()
        std = added.std() + 1e-05
        normalized = (added - mean) / std
        result = normalized * layer_norm_weight + layer_norm_bias
        return result


def replacement_func():
    return fused_layer_norm_addition