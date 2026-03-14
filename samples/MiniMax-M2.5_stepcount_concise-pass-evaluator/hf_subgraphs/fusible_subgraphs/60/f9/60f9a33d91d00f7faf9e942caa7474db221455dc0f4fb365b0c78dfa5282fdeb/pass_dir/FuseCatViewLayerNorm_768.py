import torch
import triton
import triton.language as tl


@triton.jit
def fused_cat_view_layernorm_kernel_768(
    # Input feature pointers (4 tensors)
    in_ptr2, in_ptr3, in_ptr4, in_ptr5,
    # LayerNorm parameters
    weight_ptr, bias_ptr,
    # Output pointer
    out_ptr,
    # Tensor dimensions
    N: tl.constexpr,  # hidden dimension = 768
    M: tl.constexpr,  # number of tokens
    hidden_dim: tl.constexpr,
    eps: tl.constexpr,
):
    """
    Fused kernel for hidden_dim=768 that:
    1. Concatenates 4 feature tensors along the last dimension
    2. Reshapes to (1, M, hidden_dim)
    3. Applies LayerNorm over the last dimension
    """
    # Each program processes a token (row) in the sequence
    row_idx = tl.program_id(0)
    
    # For hidden_dim=768, each tensor has 192 channels
    c_per_tensor = 192
    
    # Create offsets for loading (use power of 2: 256)
    inner_offsets = tl.arange(0, 256)
    inner_mask = inner_offsets < c_per_tensor
    
    # Load data from all 4 tensors for this row
    # Each tensor contributes 192 elements (mask the extra 64)
    x0 = tl.load(in_ptr2 + row_idx * c_per_tensor + inner_offsets, mask=inner_mask, other=0.0)
    x1 = tl.load(in_ptr3 + row_idx * c_per_tensor + inner_offsets, mask=inner_mask, other=0.0)
    x2 = tl.load(in_ptr4 + row_idx * c_per_tensor + inner_offsets, mask=inner_mask, other=0.0)
    x3 = tl.load(in_ptr5 + row_idx * c_per_tensor + inner_offsets, mask=inner_mask, other=0.0)
    
    # Compute sum and sum of squares for layer norm
    sum_x = tl.sum(x0, axis=0) + tl.sum(x1, axis=0) + tl.sum(x2, axis=0) + tl.sum(x3, axis=0)
    sum_x2 = tl.sum(x0 * x0, axis=0) + tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0) + tl.sum(x3 * x3, axis=0)
    
    # Calculate mean and variance
    mean = sum_x / N
    var = sum_x2 / N - mean * mean
    var = var + eps
    rstd = 1.0 / tl.sqrt(var)
    
    # Load weight and bias in chunks
    w0 = tl.load(weight_ptr + inner_offsets, mask=inner_mask, other=0.0)
    w1 = tl.load(weight_ptr + c_per_tensor + inner_offsets, mask=inner_mask, other=0.0)
    w2 = tl.load(weight_ptr + 2*c_per_tensor + inner_offsets, mask=inner_mask, other=0.0)
    w3 = tl.load(weight_ptr + 3*c_per_tensor + inner_offsets, mask=inner_mask, other=0.0)
    
    b0 = tl.load(bias_ptr + inner_offsets, mask=inner_mask, other=0.0)
    b1 = tl.load(bias_ptr + c_per_tensor + inner_offsets, mask=inner_mask, other=0.0)
    b2 = tl.load(bias_ptr + 2*c_per_tensor + inner_offsets, mask=inner_mask, other=0.0)
    b3 = tl.load(bias_ptr + 3*c_per_tensor + inner_offsets, mask=inner_mask, other=0.0)
    
    # Normalize and apply weight/bias
    y0 = (x0 - mean) * rstd * w0 + b0
    y1 = (x1 - mean) * rstd * w1 + b1
    y2 = (x2 - mean) * rstd * w2 + b2
    y3 = (x3 - mean) * rstd * w3 + b3
    
    # Store output at the correct positions
    out_offset = row_idx * N
    
    # Store each chunk to its correct position in the output
    tl.store(out_ptr + out_offset + inner_offsets, y0, mask=inner_mask)
    tl.store(out_ptr + out_offset + c_per_tensor + inner_offsets, y1, mask=inner_mask)
    tl.store(out_ptr + out_offset + 2*c_per_tensor + inner_offsets, y2, mask=inner_mask)
    tl.store(out_ptr + out_offset + 3*c_per_tensor + inner_offsets, y3, mask=inner_mask)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match the pattern: cat -> view -> layer_norm for hidden_dim=768
    """
    tmp_0 = in_0  # bias
    tmp_1 = in_1  # weight
    tmp_2 = torch.cat([in_2, in_3, in_4, in_5], -1)
    tmp_3 = tmp_2.view(1, -1, 768)
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), tmp_1, tmp_0, 1e-05)
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Extract arguments needed for the fused kernel
    """
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@torch.fx.wrap
def fused_kernel_wrapper_768(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Wrapper function that launches the fused Triton kernel for hidden_dim=768.
    Fuses: cat + view + layer_norm
    """
    hidden_dim = 768
    c_per_tensor = 192
    M = in_2.shape[1] * in_2.shape[2]  # 32*32 = 1024
    
    output = torch.empty((1, M, hidden_dim), device=in_2.device, dtype=in_2.dtype)
    
    grid = (M,)
    
    fused_cat_view_layernorm_kernel_768[grid](
        in_2, in_3, in_4, in_5,
        in_1,  # weight
        in_0,  # bias
        output,
        hidden_dim,
        M,
        hidden_dim,
        1e-05,
    )
    
    return output


def replacement_func():
    return fused_kernel_wrapper_768