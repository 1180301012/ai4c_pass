import torch
import triton
import triton.language as tl


# Pattern matching for: add + layer_norm fusion (16 hidden dim)
def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: in_2 + in_3, followed by layer_norm with normalized_shape=(16,)
    """
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (16,), in_1, in_0, 1e-05)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments needed for the fused add + layer_norm kernel.
    """
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_add_layer_norm_kernel_16(
    input_a_ptr,
    input_b_ptr,
    bias_ptr,
    weight_ptr,
    output_ptr,
    n_elements,
    n_positions: tl.constexpr,
    normalized_dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for layer_norm(add(a, b), (16,), weight, bias, eps)
    
    Grid: (n_positions,) where each program handles one (batch, seq) position
    Each position has 16 elements representing the hidden dimension.
    """
    pid = tl.program_id(0)
    pos_offset = pid * normalized_dim
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < normalized_dim
    
    # Load values and compute add
    x0 = tl.load(input_a_ptr + pos_offset + offsets, mask=mask, other=0.0)
    x1 = tl.load(input_b_ptr + pos_offset + offsets, mask=mask, other=0.0)
    x = x0 + x1
    
    # Compute mean using reduction
    sum_vals = tl.sum(x)
    mean = sum_vals / normalized_dim
    
    # Compute variance
    var_vals = (x - mean) * (x - mean)
    sq_sum = tl.sum(var_vals)
    var = sq_sum / normalized_dim + eps
    rstd = tl.rsqrt(var)
    
    # Load weight and bias
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Normalize and store
    out = (x - mean) * rstd * w + b
    tl.store(output_ptr + pos_offset + offsets, out, mask=mask)


@torch.fx.wrap
def fused_add_layer_norm_dispatcher_16(input_a, input_b, bias, weight, eps=1e-05):
    """
    Dispatch the fused add + layer_norm kernel for normalized_dim=16.
    """
    normalized_dim = 16
    total_elements = input_a.numel()
    n_positions = total_elements // normalized_dim
    
    # Use 16 as block size
    BLOCK_SIZE = 16
    
    output = torch.empty_like(input_a)
    
    grid = (n_positions,)
    
    fused_add_layer_norm_kernel_16[grid](
        input_a,
        input_b,
        bias,
        weight,
        output,
        total_elements,
        n_positions,
        normalized_dim,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_add_layer_norm_dispatcher_16