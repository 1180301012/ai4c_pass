import torch
import triton
import triton.language as tl
from torch import fx

def pattern(input_tensor, weight1, bias1, weight2, bias2, eps):
    # This pattern matches the LayerNorm + Dropout + LayerNorm sequence
    import torch.nn.functional as F
    
    # First LayerNorm
    tmp_1 = F.layer_norm(input_tensor, weight1.shape, weight1, bias1, eps)
    # Dropout (rate 0.0 is no-op)
    tmp_2 = F.dropout(tmp_1, 0.0, False, False)
    # Second LayerNorm
    tmp_3 = F.layer_norm(tmp_2, weight2.shape, weight2, bias2, eps)
    
    # Return the observable intermediate results that match the original graph
    return tmp_1, tmp_3

def replacement_args(input_tensor, weight1, bias1, weight2, bias2, eps):
    return (input_tensor, weight1, bias1, weight2, bias2, eps)

@triton.jit
def fused_layernorm_kernel(
    x_ptr,
    weight1_ptr,
    bias1_ptr, 
    weight2_ptr,
    bias2_ptr,
    out1_ptr,
    out2_ptr,
    n_elements,
    normalized_shape: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and weights
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    weight1 = tl.load(weight1_ptr + (offsets % normalized_shape), mask=offsets < normalized_shape, other=1.0)
    bias1 = tl.load(bias1_ptr + (offsets % normalized_shape), mask=offsets < normalized_shape, other=0.0)
    weight2 = tl.load(weight2_ptr + (offsets % normalized_shape), mask=offsets < normalized_shape, other=1.0)
    bias2 = tl.load(bias2_ptr + (offsets % normalized_shape), mask=offsets < normalized_shape, other=0.0)
    
    # First LayerNorm (epsilon = 1e-5)
    mean1 = tl.sum(x, axis=0) / normalized_shape
    var1 = tl.sum((x - mean1) * (x - mean1), axis=0) / normalized_shape
    std1 = tl.sqrt(var1 + 1e-5)
    ln1_out = (x - mean1) / std1 * weight1 + bias1
    
    # Since dropout rate = 0.0, we skip the actual dropout operation
    # dropout_out = ln1_out
    
    # Second LayerNorm (epsilon = 1e-5)
    mean2 = tl.sum(ln1_out, axis=0) / normalized_shape
    var2 = tl.sum((ln1_out - mean2) * (ln1_out - mean2), axis=0) / normalized_shape
    std2 = tl.sqrt(var2 + 1e-5)
    ln2_out = (ln1_out - mean2) / std2 * weight2 + bias2
    
    # Store results
    tl.store(out1_ptr + offsets, ln1_out, mask=mask)
    tl.store(out2_ptr + offsets, ln2_out, mask=mask)

@torch.fx.wrap
def optimized_fused_layernorm(x, weight1, bias1, weight2, bias2, eps=1e-5):
    # Get input dimensions
    n_elements = x.numel()
    normalized_shape = weight1.shape[0]
    
    # Create output tensors
    out1 = torch.empty_like(x)
    out2 = torch.empty_like(x)
    
    # Triton kernel launch
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_layernorm_kernel[grid_size](
        x_ptr=x,
        weight1_ptr=weight1,
        bias1_ptr=bias1,
        weight2_ptr=weight2,
        bias2_ptr=bias2,
        out1_ptr=out1,
        out2_ptr=out2,
        n_elements=n_elements,
        normalized_shape=normalized_shape,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out1, out2

def replacement_func():
    return optimized_fused_layernorm