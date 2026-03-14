import torch
import triton
import triton.language as tl

def pattern(in_3, in_1, in_0):
    """
    Match layer_norm followed by a single sigmoid operation pattern.
    Simplified version to avoid loading issues.
    """
    tmp_2 = torch.nn.functional.layer_norm(in_3, (256,), in_1, in_0, 1e-05)
    tmp_4 = tmp_2.sigmoid()
    return tmp_4

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@triton.jit
def optimized_sigmoid_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Optimized sigmoid with vectorized operations
    # Using a different approach for better performance
    neg_x = -x
    exp_neg_x = tl.exp(neg_x)
    out = 1.0 / (1.0 + exp_neg_x)
    
    # Store result
    tl.store(y_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_layer_norm_sigmoid(in_3, in_1, in_0):
    # Use PyTorch's highly optimized layer norm first
    layer_norm_out = torch.nn.functional.layer_norm(in_3, (256,), in_1, in_0, 1e-05)
    
    # Apply optimized sigmoid to avoid intermediate allocation if beneficial
    N = layer_norm_out.numel()
    
    # Use very large block sizes for this size input (76,800 elements)
    BLOCK_SIZE = 4096  # Even larger blocks for better GPU utilization
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(layer_norm_out)
    
    optimized_sigmoid_kernel[(num_programs,)](
        layer_norm_out,
        out,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_layer_norm_sigmoid