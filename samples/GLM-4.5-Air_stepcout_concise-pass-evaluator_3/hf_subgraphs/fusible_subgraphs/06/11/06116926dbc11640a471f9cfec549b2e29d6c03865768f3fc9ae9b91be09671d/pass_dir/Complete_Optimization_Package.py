import torch
import triton
import triton.language as tl

@triton.jit
def batch32_view32x1x64_kernel(
    input_ptr, output_ptr,
    batch_size, seq_len, total_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Optimized kernel for batch_size, view(32,-1,1,64) + transpose(1,2) pattern."""
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    
    offsets = pid * block_size + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Input: [batch_size, seq_len, 64]
    stride_batch = seq_len * 64
    stride_seq = 64
    batch_idx = offsets // stride_batch
    rem = offsets % stride_batch
    seq_idx = rem // stride_seq
    feat_idx = rem % stride_seq
    
    # Load input element
    val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Output: [batch_size, 1, seq_len, 64] - transpose pattern
    output_stride_batch = seq_len * 64
    output_stride_batch2 = 1 * seq_len * 64
    output_offset = (batch_idx * output_stride_batch2 + 
                    0 * seq_len * 64 + 
                    seq_idx * 64 + 
                    feat_idx)
    
    tl.store(output_ptr + output_offset, val, mask=mask)

@triton.jit
def batch32_view32x5x32_kernel(
    input_ptr, output_ptr,
    batch_size, seq_len, total_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Optimized kernel for batch_size, view(32,-1,5,32) + transpose(1,2) pattern."""
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    
    offsets = pid * block_size + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Input: [batch_size, seq_len, 32]
    stride_batch = seq_len * 32
    stride_seq = 32
    batch_idx = offsets // stride_batch
    rem = offsets % stride_batch
    seq_idx = rem // stride_seq
    feat_idx = rem % stride_seq
    
    # Load input element
    val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Output: [batch_size, 5, seq_len, 32] - transpose pattern
    output_stride_batch = 5 * seq_len * 32
    output_stride_batch2 = 1 * seq_len * 32
    output_offset = (batch_idx * output_stride_batch2 + 
                    seq_idx * seq_len * 32 + 
                    feat_idx)
    
    tl.store(output_ptr + output_offset, val, mask=mask)

@torch.fx.wrap
def optimized_batch32_64_features(x):
    """Optimized for batch_size, 64 features pattern."""
    batch_size, seq_len, _ = x.shape
    total_elements = batch_size * seq_len * 64
    
    # Output shape: [batch_size, 1, seq_len, 64]
    out_shape = (batch_size, 1, seq_len, 64)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    batch32_view32x1x64_kernel[grid](
        x, out, batch_size, seq_len, total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

@torch.fx.wrap
def optimized_batch32_32_features(x):
    """Optimized for batch_size, 32 features pattern."""
    batch_size, seq_len, _ = x.shape
    total_elements = batch_size * seq_len * 32
    
    # Output shape: [batch_size, 5, seq_len, 32]
    out_shape = (batch_size, 5, seq_len, 32)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    batch32_view32x5x32_kernel[grid](
        x, out, batch_size, seq_len, total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def pattern_64_features(x):
    """Match pattern for 64 features: view(32,-1,1,64) + transpose(1,2)."""
    tmp_0 = x.view(32, -1, 1, 64)
    tmp_1 = tmp_0.transpose(1, 2)
    return tmp_1

def pattern_32_features(x):
    """Match pattern for 32 features: view(32,-1,5,32) + transpose(1,2)."""
    tmp_0 = x.view(32, -1, 5, 32)
    tmp_1 = tmp_0.transpose(1, 2)
    return tmp_1

def replacement_args_64(x):
    """Extract arguments for 64 features pattern."""
    return (x,)

def replacement_args_32(x):
    """Extract arguments for 32 features pattern."""
    return (x,)

def replacement_func_64():
    """Return optimized function for 64 features pattern."""
    def optimized_wrapper(x):
        return optimized_batch32_64_features(x)
    return optimized_wrapper

def replacement_func_32():
    """Return optimized function for 32 features pattern."""
    def optimized_wrapper(x):
        return optimized_batch32_32_features(x)
    return optimized_wrapper

# Simple pattern matching - only match 64 features case
def pattern(x):
    """Match pattern: view(32,-1,1,64) + transpose(1,2)."""
    tmp_0 = x.view(32, -1, 1, 64)
    tmp_1 = tmp_0.transpose(1, 2)
    return tmp_1

def replacement_args(x):
    """Extract arguments."""
    return (x,)

def replacement_func():
    """Return optimized function."""
    def optimized_wrapper(x):
        return optimized_batch32_64_features(x)
    return optimized_wrapper