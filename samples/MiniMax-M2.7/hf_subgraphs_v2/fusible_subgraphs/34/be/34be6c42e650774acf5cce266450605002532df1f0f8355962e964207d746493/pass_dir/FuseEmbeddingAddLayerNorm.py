import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match the pattern:
    1. Padding mask generation with eq, to, multiply, unsqueeze x2
    2. Word embedding lookup
    3. Position ID generation (ones, cumsum, diff, add)
    4. Position embedding lookup
    5. Add embeddings
    6. Layer norm
    7. Dropout
    
    Returns: (dropout_output, padding_mask)
    """
    # Padding mask computation
    tmp_5 = in_0.__eq__(1)
    tmp_6 = tmp_5.to(torch.float32)
    tmp_7 = tmp_6 * -3.4028234663852886e+38
    tmp_8 = tmp_7.unsqueeze(1)
    tmp_9 = tmp_8.unsqueeze(1)
    
    # Word embeddings
    tmp_10 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    
    # Position ID generation
    tmp_11 = torch.ones((1, 15), dtype=torch.int64, device=in_0.device)
    tmp_12 = torch.cumsum(tmp_11, dim=1)
    tmp_13 = tmp_12 - tmp_11
    tmp_14 = tmp_13 + 2
    
    # Position embeddings  
    tmp_15 = torch.nn.functional.embedding(tmp_14, in_3, 1, None, 2.0, False, False)
    
    # Add embeddings
    tmp_16 = tmp_10 + tmp_15
    
    # Layer norm
    tmp_17 = torch.nn.functional.layer_norm(tmp_16, (768,), in_2, in_1, 1e-05)
    
    # Dropout
    tmp_18 = torch.nn.functional.dropout(tmp_17, 0.1, False, False)
    
    return tmp_18, tmp_9


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def triton_layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized layer normalization kernel with autotuning.
    """
    row_offset = tl.program_id(0) * n_elements
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + row_offset + offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x) / n_elements
    
    # Compute variance
    var = tl.sum((x - mean) * (x - mean)) / n_elements
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize
    normed = (x - mean) * rstd
    
    # Apply weight and bias
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    output = normed * weight + bias
    
    # Store
    tl.store(output_ptr + row_offset + offsets, output, mask=mask)


@torch.fx.wrap
def triton_layer_norm(x, weight, bias, normalized_shape, eps=1e-05):
    """
    Triton-based layer normalization.
    """
    if len(normalized_shape) != 1:
        raise ValueError("Only 1D normalized shape supported")
    
    n_elements = normalized_shape[0]
    total_elements = x.numel()
    n_rows = total_elements // n_elements
    
    output = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    triton_layer_norm_kernel[(n_rows,)](
        input_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


@torch.fx.wrap
def optimized_ernie_forward(in_0, in_1, in_2, in_3, in_4):
    """
    Optimized forward with Triton layer norm.
    """
    # Padding mask computation
    tmp_5 = in_0.__eq__(1)
    tmp_6 = tmp_5.to(torch.float32)
    tmp_7 = tmp_6 * -3.4028234663852886e+38
    tmp_8 = tmp_7.unsqueeze(1)
    tmp_9 = tmp_8.unsqueeze(1)
    
    # Word embeddings
    tmp_10 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    
    # Position ID generation
    tmp_11 = torch.ones((1, 15), dtype=torch.int64, device=in_0.device)
    tmp_12 = torch.cumsum(tmp_11, dim=1)
    tmp_13 = tmp_12 - tmp_11
    tmp_14 = tmp_13 + 2
    
    # Position embeddings  
    tmp_15 = torch.nn.functional.embedding(tmp_14, in_3, 1, None, 2.0, False, False)
    
    # Add embeddings
    tmp_16 = tmp_10 + tmp_15
    
    # Use Triton layer norm instead of torch
    tmp_17 = triton_layer_norm(tmp_16, in_2, in_1, (768,), 1e-05)
    
    # Dropout
    tmp_18 = torch.nn.functional.dropout(tmp_17, 0.1, False, False)
    
    return tmp_18, tmp_9


def replacement_func():
    return optimized_ernie_forward