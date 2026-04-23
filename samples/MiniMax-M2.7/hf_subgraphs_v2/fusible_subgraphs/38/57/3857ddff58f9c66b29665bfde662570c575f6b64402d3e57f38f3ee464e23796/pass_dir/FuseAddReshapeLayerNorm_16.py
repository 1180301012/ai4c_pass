import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_reshape_layernorm_kernel_16(
    x_ptr, y_ptr, output_ptr, norm_out_ptr,
    weight_ptr, bias_ptr,
    n_elements, hidden_dim: tl.constexpr,
    eps: tl.constexpr,
):
    """
    Fused kernel for hidden_dim=16 case:
    1. x + y (element-wise add)
    2. reshape (implicit via grid mapping)
    3. layer_norm on the reshaped tensor
    
    Assumes inputs are flattened and contiguous.
    """
    # Program id - each program handles one row of the reshaped tensor
    row_pid = tl.program_id(0)
    
    # Calculate row offsets for all elements in this row
    row_offset = row_pid * hidden_dim
    col_offsets = tl.arange(0, hidden_dim)
    x_offsets = row_offset + col_offsets
    x_mask = x_offsets < n_elements
    
    # Load bias and weight for layer norm (shape: [hidden_dim])
    weight = tl.load(weight_ptr + col_offsets)
    bias = tl.load(bias_ptr + col_offsets)
    
    # Load x and y for this row and add them
    x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
    y = tl.load(y_ptr + x_offsets, mask=x_mask, other=0.0)
    
    # Store the add result (reshaped)
    add_result = x + y
    tl.store(output_ptr + x_offsets, add_result, mask=x_mask)
    
    # Layer norm computation
    # Compute mean: sum(add_result) / hidden_dim
    sum_vals = tl.sum(add_result, axis=0)
    mean = sum_vals / hidden_dim
    
    # Compute variance: sum((x - mean)^2) / hidden_dim
    diff = add_result - mean
    sq_diff = diff * diff
    var = tl.sum(sq_diff, axis=0) / hidden_dim
    
    # Normalize and apply weight + bias
    std = tl.sqrt(var + eps)
    normalized = diff / std
    output = normalized * weight + bias
    
    # Store the normalized output
    norm_out_offset = row_pid * hidden_dim + col_offsets
    tl.store(norm_out_ptr + norm_out_offset, output, mask=x_mask)


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: add + reshape + layer_norm for hidden_dim=16
    in_0: bias (for layer norm)
    in_1: weight (for layer norm)
    in_2, in_3: tensors to add
    """
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2.reshape(-1, 16)
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (16,), in_1, in_0, 1e-05)
    return tmp_3, tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def fused_kernel_wrapper_16(in_0, in_1, in_2, in_3):
    """
    Wrapper for the fused add + reshape + layer_norm kernel (hidden_dim=16).
    """
    # Get shapes
    original_shape = in_2.shape  # [B, S, H]
    batch_size = original_shape[0]
    seq_len = original_shape[1]
    hidden_dim = 16
    
    n_rows = batch_size * seq_len
    n_elements = in_2.numel()
    
    # Flatten inputs to make them contiguous for the kernel
    x_ptr = in_2.flatten()
    y_ptr = in_3.flatten()
    
    # Allocate outputs
    output = torch.empty((n_rows, hidden_dim), dtype=in_2.dtype, device=in_2.device)
    norm_output = torch.empty((n_rows, hidden_dim), dtype=in_2.dtype, device=in_2.device)
    
    # Launch kernel
    grid = (n_rows,)
    
    fused_add_reshape_layernorm_kernel_16[grid](
        x_ptr=x_ptr,
        y_ptr=y_ptr,
        output_ptr=output,
        norm_out_ptr=norm_output,
        weight_ptr=in_1,
        bias_ptr=in_0,
        n_elements=n_elements,
        eps=1e-05,
    )
    
    return output, norm_output


def replacement_func():
    return fused_kernel_wrapper_16