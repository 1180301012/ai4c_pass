import torch
import triton
import triton.language as tl

@triton.jit
def combined_dual_branch_kernel_64_128x128(
    in_0_ptr, in_1_ptr, 
    out_1_ptr, out_3_ptr,
    batch_size, seq_len_0, seq_len_1,
    BLOCK_SIZE: tl.constexpr
):
    """Combined kernel optimizing both branches for 64 features + 128x128 pattern."""
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    total_elements = batch_size * (seq_len_1 + seq_len_0)
    num_blocks = (total_elements + block_size - 1) // block_size
    
    if pid >= num_blocks:
        return
        
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    
    # Determine which branch we're processing
    branch = offsets // seq_len_1  # 0 for in_1 (seq_len_1 elements), 1 for in_0 (seq_len_0 elements)
    local_offset = offsets % seq_len_1 if branch == 0 else offsets % seq_len_0
    
    # Branch 1: Process in_1 -> out_1 [batch_size, 1, seq_len_1, 64]
    if branch == 0 and local_offset < seq_len_1:
        batch_idx_1 = local_offset // (seq_len_1 * 64) if local_offset < batch_size * seq_len_1 else 0
        rem_1 = local_offset % (seq_len_1 * 64)
        seq_idx_1 = rem_1 // 64
        feat_idx_1 = rem_1 % 64
        
        # Input: [batch_size, seq_len_1, 64]
        input_offset_1 = (batch_idx_1 * seq_len_1 * 64 + seq_idx_1 * 64 + feat_idx_1)
        val_1 = tl.load(in_1_ptr + input_offset_1, mask=local_offset < seq_len_1, other=0.0)
        
        # Output: [batch_size, 1, seq_len_1, 64]
        output_stride_batch_1 = seq_len_1 * 64
        output_stride_batch2_1 = 1 * seq_len_1 * 64
        output_offset_1 = (batch_idx_1 * output_stride_batch2_1 + 
                          0 * seq_len_1 * 64 + 
                          seq_idx_1 * 64 + 
                          feat_idx_1)
        tl.store(out_1_ptr + output_offset_1, val_1, mask=local_offset < seq_len_1)
    
    # Branch 2: Process in_0 -> out_3 [batch_size, 64, 128, 128]  
    elif branch == 1 and local_offset < seq_len_0:
        batch_idx_0 = local_offset // (seq_len_0 * 64) if local_offset < batch_size * seq_len_0 else 0
        rem_0 = local_offset % (seq_len_0 * 64)
        seq_idx_0 = rem_0 // 64
        feat_idx_0 = rem_0 % 64
        
        # Input: [batch_size, seq_len_0, 64] -> permute to [batch_size, 64, seq_len_0]
        input_offset_0 = (batch_idx_0 * seq_len_0 * 64 + seq_idx_0 * 64 + feat_idx_0)
        val_0 = tl.load(in_0_ptr + input_offset_0, mask=local_offset < seq_len_0, other=0.0)
        
        # Output: [batch_size, 64, 128, 128] 
        h_idx_0 = seq_idx_0 // 128
        w_idx_0 = seq_idx_0 % 128
        
        output_offset_1 = (batch_idx_0 * 64 * 128 * 128 + 
                          feat_idx_0 * 128 * 128 + 
                          h_idx_0 * 128 + 
                          w_idx_0)
        tl.store(out_3_ptr + output_offset_1, val_0, mask=local_offset < seq_len_0)

@triton.jit
def combined_dual_branch_kernel_320_32x32(
    in_0_ptr, in_1_ptr, 
    out_1_ptr, out_3_ptr,
    batch_size, seq_len_0, seq_len_1,
    BLOCK_SIZE: tl.constexpr
):
    """Combined kernel optimizing both branches for 320 features + 32x32 pattern."""
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    total_elements = batch_size * (seq_len_1 + seq_len_0)
    num_blocks = (total_elements + block_size - 1) // block_size
    
    if pid >= num_blocks:
        return
        
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    
    # Determine which branch we're processing
    branch = offsets // seq_len_1  # 0 for in_1 (seq_len_1 elements), 1 for in_0 (seq_len_0 elements)
    local_offset = offsets % seq_len_1 if branch == 0 else offsets % seq_len_0
    
    # Branch 1: Process in_1 -> out_1 [batch_size, 5, seq_len_1, 32]
    if branch == 0 and local_offset < seq_len_1:
        batch_idx_1 = local_offset // (seq_len_1 * 32) if local_offset < batch_size * seq_len_1 else 0
        rem_1 = local_offset % (seq_len_1 * 32)
        seq_idx_1 = rem_1 // 32
        feat_idx_1 = rem_1 % 32
        
        # Input: [batch_size, seq_len_1, 32]
        input_offset_1 = (batch_idx_1 * seq_len_1 * 32 + seq_idx_1 * 32 + feat_idx_1)
        val_1 = tl.load(in_1_ptr + input_offset_1, mask=local_offset < seq_len_1, other=0.0)
        
        # Output: [batch_size, 5, seq_len_1, 32]
        output_stride_batch_1 = 1 * seq_len_1 * 32
        output_offset_1 = (batch_idx_1 * output_stride_batch_1 + 
                          seq_idx_1 * seq_len_1 * 32 + 
                          feat_idx_1)
        tl.store(out_1_ptr + output_offset_1, val_1, mask=local_offset < seq_len_1)
    
    # Branch 2: Process in_0 -> out_3 [batch_size, 320, 32, 32]
    elif branch == 1 and local_offset < seq_len_0:
        batch_idx_0 = local_offset // (seq_len_0 * 320) if local_offset < batch_size * seq_len_0 else 0
        rem_0 = local_offset % (seq_len_0 * 320)
        seq_idx_0 = rem_0 // 320
        feat_idx_0 = rem_0 % 320
        
        # Input: [batch_size, seq_len_0, 320] -> permute to [batch_size, 320, seq_len_0]
        input_offset_0 = (batch_idx_0 * seq_len_0 * 320 + seq_idx_0 * 320 + feat_idx_0)
        val_0 = tl.load(in_0_ptr + input_offset_0, mask=local_offset < seq_len_0, other=0.0)
        
        # Output: [batch_size, 320, 32, 32]
        h_idx_0 = seq_idx_0 // 32
        w_idx_0 = seq_idx_0 % 32
        
        output_offset_1 = (batch_idx_0 * 320 * 32 * 32 + 
                          feat_idx_0 * 32 * 32 + 
                          h_idx_0 * 32 + 
                          w_idx_0)
        tl.store(out_3_ptr + output_offset_1, val_0, mask=local_offset < seq_len_0)

@triton.jit
def combined_dual_branch_kernel_160_32x32(
    in_0_ptr, in_1_ptr, 
    out_1_ptr, out_3_ptr,
    batch_size, seq_len_0, seq_len_1,
    BLOCK_SIZE: tl.constexpr
):
    """Combined kernel optimizing both branches for 160 features + 32x32 pattern."""
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    total_elements = batch_size * (seq_len_1 + seq_len_0)
    num_blocks = (total_elements + block_size - 1) // block_size
    
    if pid >= num_blocks:
        return
        
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    
    # Determine which branch we're processing
    branch = offsets // seq_len_1  # 0 for in_1 (seq_len_1 elements), 1 for in_0 (seq_len_0 elements)
    local_offset = offsets % seq_len_1 if branch == 0 else offsets % seq_len_0
    
    # Branch 1: Process in_1 -> out_1 [batch_size, 5, seq_len_1, 32]
    if branch == 0 and local_offset < seq_len_1:
        batch_idx_1 = local_offset // (seq_len_1 * 32) if local_offset < batch_size * seq_len_1 else 0
        rem_1 = local_offset % (seq_len_1 * 32)
        seq_idx_1 = rem_1 // 32
        feat_idx_1 = rem_1 % 32
        
        # Input: [batch_size, seq_len_1, 32]
        input_offset_1 = (batch_idx_1 * seq_len_1 * 32 + seq_idx_1 * 32 + feat_idx_1)
        val_1 = tl.load(in_1_ptr + input_offset_1, mask=local_offset < seq_len_1, other=0.0)
        
        # Output: [batch_size, 5, seq_len_1, 32]
        output_stride_batch_1 = 1 * seq_len_1 * 32
        output_offset_1 = (batch_idx_1 * output_stride_batch_1 + 
                          seq_idx_1 * seq_len_1 * 32 + 
                          feat_idx_1)
        tl.store(out_1_ptr + output_offset_1, val_1, mask=local_offset < seq_len_1)
    
    # Branch 2: Process in_0 -> out_3 [batch_size, 160, 32, 32]
    elif branch == 1 and local_offset < seq_len_0:
        batch_idx_0 = local_offset // (seq_len_0 * 160) if local_offset < batch_size * seq_len_0 else 0
        rem_0 = local_offset % (seq_len_0 * 160)
        seq_idx_0 = rem_0 // 160
        feat_idx_0 = rem_0 % 160
        
        # Input: [batch_size, seq_len_0, 160] -> permute to [batch_size, 160, seq_len_0]
        input_offset_0 = (batch_idx_0 * seq_len_0 * 160 + seq_idx_0 * 160 + feat_idx_0)
        val_0 = tl.load(in_0_ptr + input_offset_0, mask=local_offset < seq_len_0, other=0.0)
        
        # Output: [batch_size, 160, 32, 32]
        h_idx_0 = seq_idx_0 // 32
        w_idx_0 = seq_idx_0 % 32
        
        output_offset_1 = (batch_idx_0 * 160 * 32 * 32 + 
                          feat_idx_0 * 32 * 32 + 
                          h_idx_0 * 32 + 
                          w_idx_0)
        tl.store(out_3_ptr + output_offset_1, val_0, mask=local_offset < seq_len_0)

@torch.fx.wrap
def combined_dual_branch_optimized_64_128x128(in_0, in_1):
    """Combined branch optimization for 64 features + 128x128 pattern."""
    batch_size, seq_len_0, _ = in_0.shape
    _, seq_len_1, _ = in_1.shape
    
    # Validate input shapes
    assert in_0.dtype == in_1.dtype == torch.float32, f"Expected float32, got {in_0.dtype} and {in_1.dtype}"
    
    # Branch 1 output: [batch_size, 1, seq_len_1, 64]
    out_1_shape = (batch_size, 1, seq_len_1, 64)
    out_1 = torch.empty(out_1_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Branch 2 output: [batch_size, 64, 128, 128] 
    out_3_shape = (batch_size, 64, 128, 128)
    out_3 = torch.empty(out_3_shape, dtype=in_0.dtype, device=in_0.device)
    
    BLOCK_SIZE = 1024
    total_elements = batch_size * (seq_len_1 + seq_len_0)
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    combined_dual_branch_kernel_64_128x128[grid](
        in_0, in_1, out_1, out_3,
        batch_size, seq_len_0, seq_len_1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out_1, out_3

@torch.fx.wrap
def combined_dual_branch_optimized_320_32x32(in_0, in_1):
    """Combined branch optimization for 320 features + 32x32 pattern."""
    batch_size, seq_len_0, _ = in_0.shape
    _, seq_len_1, _ = in_1.shape
    
    assert in_0.dtype == in_1.dtype == torch.float32, f"Expected float32, got {in_0.dtype} and {in_1.dtype}"
    
    # Branch 1 output: [batch_size, 5, seq_len_1, 32]
    out_1_shape = (batch_size, 5, seq_len_1, 32)
    out_1 = torch.empty(out_1_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Branch 2 output: [batch_size, 320, 32, 32]
    out_3_shape = (batch_size, 320, 32, 32)
    out_3 = torch.empty(out_3_shape, dtype=in_0.dtype, device=in_0.device)
    
    BLOCK_SIZE = 1024
    total_elements = batch_size * (seq_len_1 + seq_len_0)
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    combined_dual_branch_kernel_320_32x32[grid](
        in_0, in_1, out_1, out_3,
        batch_size, seq_len_0, seq_len_1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out_1, out_3

@torch.fx.wrap
def combined_dual_branch_optimized_160_32x32(in_0, in_1):
    """Combined branch optimization for 160 features + 32x32 pattern."""
    batch_size, seq_len_0, _ = in_0.shape
    _, seq_len_1, _ = in_1.shape
    
    assert in_0.dtype == in_1.dtype == torch.float32, f"Expected float32, got {in_0.dtype} and {in_1.dtype}"
    
    # Branch 1 output: [batch_size, 5, seq_len_1, 32]
    out_1_shape = (batch_size, 5, seq_len_1, 32)
    out_1 = torch.empty(out_1_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Branch 2 output: [batch_size, 160, 32, 32]
    out_3_shape = (batch_size, 160, 32, 32)
    out_3 = torch.empty(out_3_shape, dtype=in_0.dtype, device=in_0.device)
    
    BLOCK_SIZE = 1024
    total_elements = batch_size * (seq_len_1 + seq_len_0)
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    combined_dual_branch_kernel_160_32x32[grid](
        in_0, in_1, out_1, out_3,
        batch_size, seq_len_0, seq_len_1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out_1, out_3

def pattern(in_0, in_1):
    """Match the complete dual-branch pattern."""
    tmp_0 = in_1.view(32, -1, 1, 64)
    tmp_1 = tmp_0.transpose(1, 2)
    tmp_0 = None
    tmp_2 = in_0.permute(0, 2, 1)
    tmp_3 = tmp_2.reshape(32, 64, 128, 128)
    tmp_2 = None
    return tmp_1, tmp_3

def replacement_args(in_0, in_1):
    """Extract arguments for replacement."""
    # Just return the arguments needed, no conditional logic
    return (in_0, in_1)

def replacement_func():
    """Return optimized function."""
    def optimized_wrapper(in_0, in_1):
        # Determine which kernel to use based on feature dimension
        if in_0.shape[2] == 64:
            return combined_dual_branch_optimized_64_128x128(in_0, in_1)
        elif in_0.shape[2] == 320:
            return combined_dual_branch_optimized_320_32x32(in_0, in_1)
        else:  # 160 features
            return combined_dual_branch_optimized_160_32x32(in_0, in_1)
    
    return optimized_wrapper