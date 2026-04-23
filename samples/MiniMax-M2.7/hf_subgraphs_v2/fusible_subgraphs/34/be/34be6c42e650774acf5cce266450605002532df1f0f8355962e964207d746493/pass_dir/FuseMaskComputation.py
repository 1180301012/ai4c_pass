import torch
import triton
import triton.language as tl
from torch import device


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match the full ErnieM embedding pattern exactly.
    """
    # Padding mask computation (matching model's inline style)
    tmp_5 = in_0.__eq__(1)
    tmp_6 = tmp_5.to(torch.float32)
    tmp_6 *= -3.4028234663852886e+38
    tmp_7 = tmp_6
    tmp_8 = tmp_7.unsqueeze(1)
    tmp_9 = tmp_8.unsqueeze(1)
    
    # Word embeddings (matching exact model code)
    tmp_10 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    
    # Position ID generation (matching exact model code)
    tmp_11 = torch.ones((1, 15), dtype=torch.int64, device=device(type='cuda', index=0))
    tmp_12 = torch.cumsum(tmp_11, dim=1)
    tmp_13 = tmp_12 - tmp_11
    tmp_13 += 2
    tmp_14 = tmp_13
    
    # Position embeddings (matching exact model code)
    tmp_15 = torch.nn.functional.embedding(tmp_14, in_3, 1, None, 2.0, False, False)
    
    # Add embeddings
    tmp_16 = tmp_10 + tmp_15
    
    # Layer norm (matching exact model code with hardcoded (768,))
    tmp_17 = torch.nn.functional.layer_norm(tmp_16, (768,), in_2, in_1, 1e-05)
    
    # Dropout (matching exact model code)
    tmp_18 = torch.nn.functional.dropout(tmp_17, 0.1, False, False)
    
    return tmp_18, tmp_9


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def triton_mask_kernel(
    in_0_ptr,
    mask_out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized padding mask kernel using Triton.
    Computes: (in_0 == 1).to(float32) * -3.4028234663852886e+38
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(in_0_ptr + offsets, mask=mask, other=0)
    
    # Compute: (x == 1).to(float32) * -3.4028234663852886e+38
    is_one = tl.where(x == 1, 1.0, 0.0)
    result = is_one * -3.4028234663852886e+38
    
    # Store
    tl.store(mask_out_ptr + offsets, result, mask=mask)


@triton.jit
def triton_add_layernorm_kernel(
    a_ptr,
    b_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused add + layer norm kernel.
    """
    row_offset = tl.program_id(0) * n_elements
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    a = tl.load(a_ptr + row_offset + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + row_offset + offsets, mask=mask, other=0.0)
    
    # Add
    x = a + b
    
    # Layer norm
    mean = tl.sum(x) / n_elements
    var = tl.sum((x - mean) * (x - mean)) / n_elements
    rstd = 1.0 / tl.sqrt(var + eps)
    normed = (x - mean) * rstd
    
    # Weight and bias
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    output = normed * weight + bias
    
    # Store
    tl.store(output_ptr + row_offset + offsets, output, mask=mask)


@torch.fx.wrap
def triton_mask(in_0):
    """Triton-based padding mask computation."""
    n_elements = in_0.numel()
    mask_output = torch.empty_like(in_0, dtype=torch.float32)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    triton_mask_kernel[(num_programs,)](
        in_0_ptr=in_0,
        mask_out_ptr=mask_output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return mask_output


@torch.fx.wrap
def triton_add_layernorm(a, b, weight, bias, normalized_shape, eps=1e-05):
    """Triton-based fused add + layer norm."""
    if len(normalized_shape) != 1:
        raise ValueError("Only 1D normalized shape supported")
    
    n_elements = normalized_shape[0]
    total_elements = a.numel()
    n_rows = total_elements // n_elements
    
    output = torch.empty_like(a)
    
    BLOCK_SIZE = 1024
    triton_add_layernorm_kernel[(n_rows,)](
        a_ptr=a,
        b_ptr=b,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    pass