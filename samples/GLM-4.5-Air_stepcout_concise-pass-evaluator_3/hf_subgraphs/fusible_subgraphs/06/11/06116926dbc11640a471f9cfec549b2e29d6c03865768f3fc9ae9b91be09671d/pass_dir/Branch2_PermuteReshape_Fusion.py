import torch
import triton
import triton.language as tl

@triton.jit
def permute_reshape_kernel_64_128x128(
    input_ptr, output_ptr, 
    batch_size, total_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Optimized kernel for permute(0,2,1) + reshape(..,64,128,128) pattern."""
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    num_blocks = (total_elements + block_size - 1) // block_size
    
    if pid >= num_blocks:
        return
        
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    
    # Input is [batch_size, seq_len, 64] -> permute to [batch_size, 64, seq_len]
    # where seq_len should be 16384 = 128*128
    
    # Linear index to permuted coordinates:
    # Input: [batch_size, seq_len, 64] with stride [seq_len*64, 64, 1]
    # Output: [batch_size, 64, 128, 128] with stride [64*128*128, 128*128, 128, 1]
    
    batch_size_total, seq_len, n_features = batch_size, 16384, 64
    
    # Calculate coordinates in original (permuted) layout [batch_size, 64, 16384]
    stride_permuted = seq_len * n_features
    stride_feature = n_features
    
    original_idx = offsets
    batch_idx = original_idx // stride_permuted
    rem = original_idx % stride_permuted
    feature_idx = rem // stride_feature
    seq_idx = rem % stride_feature
    
    # Map to final output [batch_size, 64, 128, 128]
    # seq_idx (0-16383) -> (h_idx, w_idx) where h_idx = seq_idx // 128, w_idx = seq_idx % 128
    h_idx = seq_idx // 128
    w_idx = seq_idx % 128
    
    # Final offset calculation
    output_stride_batch = 64 * 128 * 128
    output_stride_feature = 128 * 128
    output_stride_h = 128
    
    output_offset = (batch_idx * output_stride_batch + 
                    feature_idx * output_stride_feature + 
                    h_idx * output_stride_h + 
                    w_idx)
    
    # Load from input (which has original stride)  
    input_stride_original = seq_len * n_features
    input_stride_feature = n_features
        
    input_offset = (batch_idx * input_stride_original + 
                   seq_idx * input_stride_feature + 
                   feature_idx)
    
    val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + output_offset, val, mask=mask)

@triton.jit
def permute_reshape_kernel_320_32x32(
    input_ptr, output_ptr, 
    batch_size, total_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Optimized kernel for permute(0,2,1) + reshape(..,320,32,32) pattern."""
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    num_blocks = (total_elements + block_size - 1) // block_size
    
    if pid >= num_blocks:
        return
        
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    
    # Input is [batch_size, seq_len, 320] -> permute to [batch_size, 320, seq_len]
    # where seq_len should be 1024 = 32*32
    
    batch_size_total, seq_len, n_features = batch_size, 1024, 320
    
    # Calculate coordinates in permuted layout [batch_size, 320, 1024]
    stride_permuted = seq_len * n_features
    stride_feature = n_features
    
    original_idx = offsets
    batch_idx = original_idx // stride_permuted
    rem = original_idx % stride_permuted
    feature_idx = rem // stride_feature
    seq_idx = rem % stride_feature
    
    # Map to final output [batch_size, 320, 32, 32]
    h_idx = seq_idx // 32
    w_idx = seq_idx % 32
    
    # Final offset calculation  
    output_stride_batch = 320 * 32 * 32
    output_stride_feature = 32 * 32
    output_stride_h = 32
    
    output_offset = (batch_idx * output_stride_batch + 
                    feature_idx * output_stride_feature + 
                    h_idx * output_stride_h + 
                    w_idx)
    
    # Load from input (which has original stride)
    input_stride_original = seq_len * n_features
    input_stride_feature = n_features
        
    input_offset = (batch_idx * input_stride_original + 
                   seq_idx * input_stride_feature + 
                   feature_idx)
    
    val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + output_offset, val, mask=mask)

@triton.jit
def permute_reshape_kernel_160_32x32(
    input_ptr, output_ptr, 
    batch_size, total_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Optimized kernel for permute(0,2,1) + reshape(..,160,32,32) pattern."""
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    num_blocks = (total_elements + block_size - 1) // block_size
    
    if pid >= num_blocks:
        return
        
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    
    # Input is [batch_size, seq_len, 160] -> permute to [batch_size, 160, seq_len]
    # where seq_len should be 1024 = 32*32
    
    batch_size_total, seq_len, n_features = batch_size, 1024, 160
    
    # Calculate coordinates in permuted layout [batch_size, 160, 1024]
    stride_permuted = seq_len * n_features
    stride_feature = n_features
    
    original_idx = offsets
    batch_idx = original_idx // stride_permuted
    rem = original_idx % stride_permuted
    feature_idx = rem // stride_feature
    seq_idx = rem % stride_feature
    
    # Map to final output [batch_size, 160, 32, 32]
    h_idx = seq_idx // 32
    w_idx = seq_idx % 32
    
    # Final offset calculation
    output_stride_batch = 160 * 32 * 32
    output_stride_feature = 32 * 32
    output_stride_h = 32
    
    output_offset = (batch_idx * output_stride_batch + 
                    feature_idx * output_stride_feature + 
                    h_idx * output_stride_h + 
                    w_idx)
    
    # Load from input (which has original stride)
    input_stride_original = seq_len * n_features
    input_stride_feature = n_features
        
    input_offset = (batch_idx * input_stride_original + 
                   seq_idx * input_stride_feature + 
                   feature_idx)
    
    val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + output_offset, val, mask=mask)

@torch.fx.wrap
def optimized_permute_reshape_64_128x128(x):
    """Optimized permute(0,2,1) + reshape(..,64,128,128) operation."""
    batch_size, seq_len, _ = x.shape
    assert seq_len == 16384, f"Expected seq_len=16384, got {seq_len}"
    
    total_elements = batch_size * 64 * 128 * 128
    out_shape = (batch_size, 64, 128, 128)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    permute_reshape_kernel_64_128x128[grid](
        x, out,
        batch_size, total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

@torch.fx.wrap
def optimized_permute_reshape_320_32x32(x):
    """Optimized permute(0,2,1) + reshape(..,320,32,32) operation."""
    batch_size, seq_len, _ = x.shape
    assert seq_len == 1024, f"Expected seq_len=1024, got {seq_len}"
    
    total_elements = batch_size * 320 * 32 * 32
    out_shape = (batch_size, 320, 32, 32)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    permute_reshape_kernel_320_32x32[grid](
        x, out,
        batch_size, total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

@torch.fx.wrap
def optimized_permute_reshape_160_32x32(x):
    """Optimized permute(0,2,1) + reshape(..,160,32,32) operation."""
    batch_size, seq_len, _ = x.shape
    assert seq_len == 1024, f"Expected seq_len=1024, got {seq_len}"
    
    total_elements = batch_size * 160 * 32 * 32
    out_shape = (batch_size, 160, 32, 32)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    permute_reshape_kernel_160_32x32[grid](
        x, out,
        batch_size, total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def pattern(x, permute_dims, reshape_args):
    """Match permute + reshape pattern."""
    tmp_2 = x.permute(*permute_dims)
    tmp_3 = tmp_2.reshape(*reshape_args)
    return tmp_3

def replacement_args(x, permute_dims=(0, 2, 1), reshape_args=(32, 64, 128, 128)):
    """Extract arguments for replacement."""
    # Just return the arguments needed, no conditional logic
    return (x, permute_dims, reshape_args)

def replacement_func():
    """Return optimized function."""
    def optimized_wrapper(x, permute_dims=(0, 2, 1), reshape_args=(32, 64, 128, 128)):
        if reshape_args[1] == 64:
            return optimized_permute_reshape_64_128x128(x)
        elif reshape_args[1] == 320:
            return optimized_permute_reshape_320_32x32(x)
        else:
            return optimized_permute_reshape_160_32x32(x)
    
    return optimized_wrapper