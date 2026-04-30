import torch
import triton
import triton.language as tl

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
    Autotuned kernel for fused add + reshape + layer_norm with hidden_size=16.
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
    Pattern: Fuse add + reshape + layer_norm for hidden_size=16
    in_0: bias tensor [16]
    in_1: weight tensor [16]
    in_2: input tensor [batch, seq, 16]
    in_3: input tensor [batch, seq, 16]
    """
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2.reshape(-1, 16)
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (16,), in_1, in_0, 1e-05)
    return tmp_3, tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_add_reshape_layernorm_wrapper