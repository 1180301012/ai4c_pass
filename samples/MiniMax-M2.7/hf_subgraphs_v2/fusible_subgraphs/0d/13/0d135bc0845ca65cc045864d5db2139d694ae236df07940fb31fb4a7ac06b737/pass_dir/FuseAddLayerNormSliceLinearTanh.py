import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_layernorm_kernel(
    in_5_ptr, in_6_ptr,
    in_2_ptr, in_1_ptr,  # Note: in_2 is weight (gamma), in_1 is bias (beta)
    tmp_6_ptr,
    stride_in_5_0, stride_in_5_1, stride_in_5_2,
    stride_in_6_0, stride_in_6_1, stride_in_6_2,
    stride_tmp_6_0, stride_tmp_6_1,
    n_elements_batch, n_elements_seq, n_elements_hidden,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: add + layer_norm
    
    Computes:
    1. tmp_5 = in_6 + in_5  (element-wise add)
    2. tmp_6 = layer_norm(tmp_5, (384,), in_2, in_1, 1e-12)
       where in_2 is weight (gamma) and in_1 is bias (beta)
    
    Output:
    - tmp_6 (full normalized tensor)
    """
    
    # Get batch and sequence positions
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    
    # Calculate base offset for this batch/sequence position
    base_offset = pid_batch * stride_in_5_0 + pid_seq * stride_in_5_1
    
    # Calculate number of hidden elements
    n_hidden = n_elements_hidden
    
    # Create offsets for hidden dimension - BLOCK_SIZE must be power of 2
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_hidden
    
    # Pre-compute gamma and beta offsets (1D tensor, stride=1)
    param_offs = offs.to(tl.int64)
    
    # Load gamma and beta once (they're the same for all programs)
    gamma = tl.load(in_2_ptr + param_offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(in_1_ptr + param_offs, mask=mask, other=0.0).to(tl.float32)
    
    # Compute flat offsets for input tensors (may have different strides)
    in_5_offsets = (base_offset + offs * stride_in_5_2).to(tl.int64)
    in_6_offsets = (base_offset + offs * stride_in_6_2).to(tl.int64)
    
    # Load input values
    in_5_vals = tl.load(in_5_ptr + in_5_offsets, mask=mask, other=0.0).to(tl.float32)
    in_6_vals = tl.load(in_6_ptr + in_6_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute tmp_5 = in_6 + in_5
    tmp_5_vals = in_6_vals + in_5_vals
    
    # Layer Norm: compute mean and variance
    sum_vals = tl.sum(tmp_5_vals, axis=0)
    mean = sum_vals / n_hidden
    
    diff_sq = (tmp_5_vals - mean) * (tmp_5_vals - mean)
    sum_var = tl.sum(diff_sq, axis=0)
    var = sum_var / n_hidden
    
    # Normalize and apply affine transform
    inv_std = tl.rsqrt(var + eps)
    normalized = (tmp_5_vals - mean) * inv_std
    tmp_6_vals = normalized * gamma + beta
    
    # Store to output buffer (contiguous)
    tmp_6_base = pid_batch * stride_tmp_6_0 + pid_seq * stride_tmp_6_1
    tmp_6_offsets = (tmp_6_base + offs).to(tl.int64)
    tl.store(tmp_6_ptr + tmp_6_offsets, tmp_6_vals, mask=mask)


@torch.fx.wrap
def fused_add_layernorm_wrapper(in_5, in_6, in_1, in_2):
    """
    Wrapper for the fused kernel that computes:
    tmp_5 = in_6 + in_5
    tmp_6 = layer_norm(tmp_5, (384,), in_2, in_1, 1e-12)
    Returns: tmp_6
    """
    batch_size, seq_len, hidden_dim = in_5.shape
    device = in_5.device
    
    # tmp_6 shape: (batch_size, seq_len, hidden_dim)
    tmp_6 = torch.empty((batch_size, seq_len, hidden_dim), device=device, dtype=in_5.dtype)
    
    # Define grid
    grid = (batch_size, seq_len)
    
    # Must be power of 2 for Triton
    BLOCK_SIZE = 512
    
    # Pass arguments in order matching kernel signature:
    # (in_5, in_6, in_2, in_1, ...) - in_2 is weight, in_1 is bias
    fused_add_layernorm_kernel[grid](
        in_5, in_6,
        in_2, in_1,  # weight (gamma), bias (beta)
        tmp_6,
        in_5.stride(0), in_5.stride(1), in_5.stride(2),
        in_6.stride(0), in_6.stride(1), in_6.stride(2),
        tmp_6.stride(0), tmp_6.stride(1),
        batch_size, seq_len, hidden_dim,
        eps=1e-12,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return tmp_6


def pattern(in_5, in_6, in_1, in_2):
    """
    Match the add + layer_norm pattern:
    tmp_5 = in_6 + in_5
    tmp_6 = layer_norm(tmp_5, (384,), in_2, in_1, 1e-12)
    Returns: tmp_6
    """
    tmp_5 = in_6 + in_5
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (384,), in_2, in_1, 1e-12)
    return tmp_6


def replacement_args(in_5, in_6, in_1, in_2):
    return (in_5, in_6, in_1, in_2)


def replacement_func():
    return fused_add_layernorm_wrapper