import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Pattern that matches max + expand + subtraction for softmax numerical stability"""
    max_1 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = max_1[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    return tmp_3

def replacement_args(in_0):
    """Extract arguments for the replacement function"""
    return (in_0,)

# Define kernels for different data types
@triton.jit
def fused_max_sub_kernel_float32(x_ptr, out_ptr, n_batch, n_features, n_elements_last_dim, BLOCK_SIZE_N: tl.constexpr):
    """Fused max and subtraction kernel for float32"""
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)

    batch_features_offset = batch_idx * n_features * n_elements_last_dim + feature_idx * n_elements_last_dim
    x_ptr_batch = x_ptr + batch_features_offset
    out_ptr_batch = out_ptr + batch_features_offset

    offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < n_elements_last_dim
    x = tl.load(x_ptr_batch + offsets, mask=mask, other=-float('inf'))
    
    local_max = tl.max(x)
    result = x - local_max
    
    tl.store(out_ptr_batch + offsets, result, mask=mask)

@triton.jit
def fused_max_sub_kernel_float16(x_ptr, out_ptr, n_batch, n_features, n_elements_last_dim, BLOCK_SIZE_N: tl.constexpr):
    """Fused max and subtraction kernel for float16"""
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)

    batch_features_offset = batch_idx * n_features * n_elements_last_dim + feature_idx * n_elements_last_dim
    x_ptr_batch = x_ptr + batch_features_offset
    out_ptr_batch = out_ptr + batch_features_offset

    offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < n_elements_last_dim
    x = tl.load(x_ptr_batch + offsets, mask=mask, other=-float('inf'))
    
    local_max = tl.max(x)
    result = x - local_max
    
    tl.store(out_ptr_batch + offsets, result, mask=mask)

@triton.jit
def fused_max_sub_kernel_bfloat16(x_ptr, out_ptr, n_batch, n_features, n_elements_last_dim, BLOCK_SIZE_N: tl.constexpr):
    """Fused max and subtraction kernel for bfloat16"""
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)

    batch_features_offset = batch_idx * n_features * n_elements_last_dim + feature_idx * n_elements_last_dim
    x_ptr_batch = x_ptr + batch_features_offset
    out_ptr_batch = out_ptr + batch_features_offset

    offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < n_elements_last_dim
    x = tl.load(x_ptr_batch + offsets, mask=mask, other=-float('inf'))
    
    local_max = tl.max(x)
    result = x - local_max
    
    tl.store(out_ptr_batch + offsets, result, mask=mask)

@torch.fx.wrap
def fused_max_sub_torch(in_0):
    """PyTorch wrapper for the fused max-subtraction operation"""
    if in_0.dim() != 3:
        raise ValueError(f"Expected 3D input, got {in_0.dim()}D")
    
    n_batch, n_features, n_elements_last_dim = in_0.shape
    
    # Choose kernel based on data type
    if in_0.dtype == torch.float32:
        kernel = fused_max_sub_kernel_float32
        BLOCK_SIZE_N = 1024
    elif in_0.dtype == torch.float16:
        kernel = fused_max_sub_kernel_float16
        BLOCK_SIZE_N = 1024
    elif in_0.dtype == torch.bfloat16:
        kernel = fused_max_sub_kernel_bfloat16
        BLOCK_SIZE_N = 1024
    else:
        raise ValueError(f"Unsupported dtype: {in_0.dtype}")
    
    out = torch.empty_like(in_0)
    
    # Launch kernel
    grid = (
        (n_batch + 31) // 32,
        (n_features + 31) // 32,
    )
    
    kernel[grid](
        in_0,
        out,
        n_batch,
        n_features,
        n_elements_last_dim,
        BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    """Return the optimized function"""
    return fused_max_sub_torch