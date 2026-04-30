import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_reshape_layernorm_kernel(
    x1_ptr,
    x2_ptr,
    weight_ptr,
    bias_ptr,
    out_reshaped_ptr,
    out_norm_ptr,
    n_elements,
    normalized_shape,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Element-wise addition of x1 and x2
    2. Reshape from [batch, seq, hidden] to [batch*seq, hidden]
    3. Layer normalization over the hidden dimension
    
    Args:
        x1_ptr: First input tensor pointer [batch, seq, hidden]
        x2_ptr: Second input tensor pointer [batch, seq, hidden]
        weight_ptr: Layer norm weight pointer [hidden]
        bias_ptr: Layer norm bias pointer [hidden]
        out_reshaped_ptr: Output for reshaped tensor [batch*seq, hidden]
        out_norm_ptr: Output for normalized tensor [batch*seq, hidden]
        n_elements: Total number of elements (batch * seq * hidden)
        normalized_shape: The hidden dimension size
        eps: Epsilon for numerical stability
        BLOCK_SIZE: Threads per block
    """
    # Get program IDs
    pid = tl.program_id(0)
    
    # Calculate block offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x1 and x2
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Element-wise addition
    x = x1 + x2
    
    # Store reshaped output
    tl.store(out_reshaped_ptr + offsets, x, mask=mask)
    
    # For layer norm, we need to compute mean and variance per row
    # Each program processes multiple elements in a row
    # We need to compute mean: sum(x) / n_elements_per_row
    # Then variance: sum((x - mean)^2) / n_elements_per_row
    
    # Calculate which "row" this element belongs to (row = hidden dimension index)
    row_offset = offsets % normalized_shape
    row_mask = row_offset < normalized_shape
    
    # Load data for layer norm computation
    x_for_norm = x
    
    # Compute partial sum and sum of squares for mean and variance
    # Use a reduction approach
    sum_vals = x_for_norm
    sq_vals = x_for_norm * x_for_norm
    
    # Thread-local computation for now - for better accuracy we should use block reduction
    # but for simplicity, compute per-element approximation
    # Actually for proper layer norm, we need full row statistics
    # Let's reload the entire row
    
    # For proper layer norm, we need to compute mean and variance per row
    # The row is defined by offsets // normalized_shape
    row_id = offsets // normalized_shape
    
    # Each thread computes mean as sum of its portion of the row
    # We'll use a simpler approach: each thread loads its value and we'll do reduction
    mean = x_for_norm
    
    # Variance computation
    variance = (x_for_norm - mean) * (x_for_norm - mean)
    
    # Normalize
    std = tl.sqrt(variance + eps)
    normed = (x_for_norm - mean) / std
    
    # Apply affine transform
    w = tl.load(weight_ptr + row_offset, mask=row_mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + row_offset, mask=row_mask, other=0.0).to(tl.float32)
    
    out = normed * w + b
    
    # Store normalized output
    tl.store(out_norm_ptr + offsets, out, mask=mask)


@triton.jit
def fused_add_reshape_layernorm_kernel_v2(
    x1_ptr,
    x2_ptr,
    weight_ptr,
    bias_ptr,
    out_reshaped_ptr,
    out_norm_ptr,
    n_rows,
    normalized_shape,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized v2: Properly compute mean and variance per row using reduction.
    
    Grid: (n_rows,) - one program per row
    Each program computes one row's layer norm.
    """
    row_id = tl.program_id(0)
    
    # Base offset for this row
    row_start = row_id * normalized_shape
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_id + 1) * normalized_shape
    
    # Load data
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Element-wise addition
    x = x1 + x2
    
    # Store reshaped output
    tl.store(out_reshaped_ptr + offsets, x, mask=mask)
    
    # Compute sum using reduction
    sum_val = tl.sum(x, axis=0)
    mean = sum_val / normalized_shape
    
    # Compute variance
    diff = x - mean
    sq_diff = diff * diff
    var_sum = tl.sum(sq_diff, axis=0)
    variance = var_sum / normalized_shape
    std = tl.sqrt(variance + eps)
    
    # Normalize
    normed = diff / std
    
    # Apply affine transform
    w_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    w_mask = w_offsets < (row_id + 1) * normalized_shape
    w = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=w_mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=w_mask, other=0.0).to(tl.float32)
    
    out = normed * w + b
    
    # Store output
    tl.store(out_norm_ptr + offsets, out, mask=mask)


# Autotune configuration
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
    ],
    key=['n_rows'],
)
@triton.jit
def fused_add_reshape_layernorm_kernel_autotuned(
    x1_ptr,
    x2_ptr,
    weight_ptr,
    bias_ptr,
    out_reshaped_ptr,
    out_norm_ptr,
    n_rows,
    normalized_shape,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Autotuned kernel for fused add + reshape + layer_norm.
    One program per row (batch*seq).
    """
    row_id = tl.program_id(0)
    
    # Base offset for this row
    row_start = row_id * normalized_shape
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_id + 1) * normalized_shape
    
    # Load data
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Element-wise addition
    x = x1 + x2
    
    # Store reshaped output
    tl.store(out_reshaped_ptr + offsets, x, mask=mask)
    
    # Compute sum using reduction
    sum_val = tl.sum(x, axis=0)
    mean = sum_val / normalized_shape
    
    # Compute variance
    diff = x - mean
    sq_diff = diff * diff
    var_sum = tl.sum(sq_diff, axis=0)
    variance = var_sum / normalized_shape
    std = tl.sqrt(variance + eps)
    
    # Normalize
    normed = diff / std
    
    # Apply affine transform
    w = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(tl.float32)
    
    out = normed * w + b
    
    # Store output
    tl.store(out_norm_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_add_reshape_layernorm_wrapper(x1, x2, weight, bias, normalized_shape, eps=1e-05):
    """
    Wrapper function to call the fused kernel.
    
    Args:
        x1: Input tensor [batch, seq, hidden] - in_2
        x2: Input tensor [batch, seq, hidden] - in_3
        weight: Layer norm weight [hidden] - in_1
        bias: Layer norm bias [hidden] - in_0
        normalized_shape: Tuple specifying normalized dimension
        eps: Epsilon for numerical stability
    
    Returns:
        reshaped: x1 + x2 reshaped to [batch*seq, hidden]
        normalized: Layer norm output [batch*seq, hidden]
    """
    batch, seq, hidden = x1.shape
    
    # Compute output shapes
    n_rows = batch * seq
    n_elements = n_rows * hidden
    
    # Allocate output tensors with same dtype as input
    dtype = x1.dtype
    reshaped = torch.empty((n_rows, hidden), dtype=dtype, device=x1.device)
    normalized = torch.empty((n_rows, hidden), dtype=dtype, device=x1.device)
    
    # Flatten input tensors for kernel
    x1_flat = x1.view(-1)
    x2_flat = x2.view(-1)
    
    # Launch kernel - one program per row
    grid = (n_rows,)
    
    fused_add_reshape_layernorm_kernel_autotuned[grid](
        x1_flat,
        x2_flat,
        weight,
        bias,
        reshaped,
        normalized,
        n_rows,
        hidden,
        eps,
    )
    
    return reshaped, normalized


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern: Fuse add + reshape + layer_norm
    in_0: bias tensor [hidden]
    in_1: weight tensor [hidden]
    in_2: input tensor [batch, seq, hidden]
    in_3: input tensor [batch, seq, hidden]
    """
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2.reshape(-1, 768)
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), in_1, in_0, 1e-05)
    return tmp_3, tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_add_reshape_layernorm_wrapper