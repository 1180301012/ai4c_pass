import torch
import triton
import triton.language as tl

# Pattern matching for attention-style matmul: (B,H,O,I) @ (B,H,I,1) -> (B,H,O,1) then view to (B,H*O,1,1)
def pattern(a, b):
    """Match attention-style matmul followed by view operation"""
    result = a @ b
    return result

def replacement_args(a, b):
    """Extract arguments for the replacement kernel"""
    return (a, b)

# Highly optimized Triton kernel for attention-style operations
@triton.jit
def attention_matmul_kernel_fp16(
    a_ptr, b_ptr, output_ptr,
    batch_heads, output_features, input_seq,
    BLOCK_SIZE_O: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
):
    """Optimized kernel for attention-style batched matmul: (BH,O,I) @ (BH,I,1) -> (BH,O,1)"""
    
    # Program ID per batch-head
    bh_pid = tl.program_id(0)
    
    # O and I offsets for this thread
    o_offset = bh_pid * output_features + tl.arange(0, BLOCK_SIZE_O)
    i_offset = tl.arange(0, BLOCK_SIZE_I)
    
    # Create mask for bounds checking
    o_mask = o_offset < output_features
    i_mask = i_offset < input_seq
    
    # Initialize output for this block
    output_block = tl.zeros((BLOCK_SIZE_O, 1), dtype=tl.float32)
    
    # Compute matmul for current block
    if o_mask.any() and i_mask.any():
        # Load A matrix block (BH,O,I)
        a_ptrs = a_ptr + bh_pid * output_features * input_seq + o_offset[:, None] * input_seq + i_offset[None, :]
        a_block = tl.load(a_ptrs, mask=o_mask[:, None] & i_mask[None, :], other=0.0)
        
        # Load B matrix block (BH,I,1)
        b_ptrs = b_ptr + bh_pid * input_seq * 1 + i_offset[:, None] * 1 + 0
        b_block = tl.load(b_ptrs, mask=i_mask[:, None] & (tl.arange(0, 1) < 1), other=0.0)
        
        # Perform matrix multiplication: (O,I) @ (I,1) -> (O,1)
        result = tl.dot(a_block, b_block.to(tl.float32), out_dtype=tl.float32)
        output_block = result
    
    # Store result at the correct position
    output_base = bh_pid * output_features * 1
    output_ptrs = output_ptr + output_base + o_offset[:, None] * 1 + 0
    
    tl.store(output_ptrs, output_block.to(tl.float16), mask=o_mask[:, None] & (tl.arange(0, 1) < 1))

@triton.jit
def attention_matmul_kernel_bf16(
    a_ptr, b_ptr, output_ptr,
    batch_heads, output_features, input_seq,
    BLOCK_SIZE_O: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
):
    """BF16 optimized version of attention kernel"""
    
    bh_pid = tl.program_id(0)
    
    o_offset = bh_pid * output_features + tl.arange(0, BLOCK_SIZE_O)
    i_offset = tl.arange(0, BLOCK_SIZE_I)
    
    o_mask = o_offset < output_features
    i_mask = i_offset < input_seq
    
    output_block = tl.zeros((BLOCK_SIZE_O, 1), dtype=tl.bfloat16)
    
    if o_mask.any() and i_mask.any():
        a_ptrs = a_ptr + bh_pid * output_features * input_seq + o_offset[:, None] * input_seq + i_offset[None, :]
        a_block = tl.load(a_ptrs, mask=o_mask[:, None] & i_mask[None, :], other=0.0)
        
        b_ptrs = b_ptr + bh_pid * input_seq * 1 + i_offset[:, None] * 1 + 0
        b_block = tl.load(b_ptrs, mask=i_mask[:, None] & (tl.arange(0, 1) < 1), other=0.0)
        
        # Use BF16 dot product
        result = tl.dot(a_block, b_block, out_dtype=tl.bfloat16)
        output_block = result
    
    output_base = bh_pid * output_features * 1
    output_ptrs = output_ptr + output_base + o_offset[:, None] * 1 + 0
    
    tl.store(output_ptrs, output_block, mask=o_mask[:, None] & (tl.arange(0, 1) < 1))

# Optimized kernel wrapper for attention-style operations
@torch.fx.wrap
def optimized_attention_matmul(a, b, target_view_shape):
    """High-performance attention-style matmul with view fusion
    
    Args:
        a: [B, H, O, I] - query/values tensor
        b: [B, H, I, 1] - keys/context tensor  
        target_view_shape: [B, H*O, 1, 1] - final view shape
    """
    # Validate input shapes
    assert len(a.shape) == 4, f"Expected 4D tensor for 'a', got {a.shape}"
    assert len(b.shape) == 4, f"Expected 4D tensor for 'b', got {b.shape}"
    assert b.shape[-1] == 1, f"Expected last dimension of 'b' to be 1, got {b.shape}"
    assert a.shape[0] == b.shape[0], f"Batch size mismatch: {a.shape[0]} vs {b.shape[0]}"
    assert a.shape[1] == b.shape[1], f"Head count mismatch: {a.shape[1]} vs {b.shape[1]}"
    assert a.shape[-1] == b.shape[-2], f"Input sequence mismatch: {a.shape[-1]} vs {b.shape[-2]}"
    
    B, H, O, I = a.shape
    _, _, _, _ = b.shape
    
    # Combine batch and heads for efficient computation
    batch_heads = B * H
    total_output_elements = batch_heads * O * 1
    
    # Select kernel based on data type
    if a.dtype == torch.bfloat16:
        kernel = attention_matmul_kernel_bf16
        dtype = torch.bfloat16
    else:
        kernel = attention_matmul_kernel_fp16
        dtype = torch.float16
    
    # Create output tensor
    output_shape = (batch_heads, O, 1)
    output = torch.empty(output_shape, dtype=dtype, device=a.device)
    
    # Configure optimal block sizes
    if O > 256:
        block_size_o = 64
    elif O > 128:
        block_size_o = 32
    else:
        block_size_o = min(O, 16)
    
    if I > 256:
        block_size_i = 32
    elif I > 128:
        block_size_i = 16
    else:
        block_size_i = min(I, 8)
    
    # Launch kernel
    grid_size = batch_heads
    kernel[grid_size](
        a, b, output,
        batch_heads, O, I,
        block_size_o, block_size_i
    )
    
    # Apply view operation: [BH,O,1] -> [B,H,O,1] -> [B,H×O,1,1]
    if target_view_shape is not None and len(target_view_shape) == 4:
        B_out, HO_out, H_out, W_out = target_view_shape
        
        # Reshape to separate batch and heads: [B,H,O,1]
        output = output.reshape(B, H, O, 1)
        
        # Combine heads and output features: [B,H×O,1,1]
        output = output.reshape(B, H * O, 1, 1)
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    def optimized_func(a, b):
        # Infer target view from attention pattern
        B, H, O, I = a.shape
        target_view = (B, H * O, 1, 1)
        return optimized_attention_matmul(a, b, target_view)
    
    return optimized_func