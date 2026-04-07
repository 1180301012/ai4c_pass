import torch
import triton
import triton.language as tl

# Pattern matching for YOLO-style matmul: (B,C,O,I) @ (B,C,I,H,W) -> (B,C,O,H,W) then view to (B,C×O,H,W)
def pattern(a, b):
    """Match YOLO-style matmul followed by view operation"""
    result = a @ b
    return result

def replacement_args(a, b):
    """Extract arguments for the replacement kernel"""
    return (a, b)

# Highly optimized Triton kernel for YOLO-style operations
@triton.jit
def yolo_matmul_kernel_fp16(
    a_ptr, b_ptr, output_ptr,
    batch_channels, output_features, input_features, height, width,
    BLOCK_SIZE_BC: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    """Optimized kernel for YOLO-style batched matmul: (BC,O,I) @ (BC,I,H*W) -> (BC,O,H*W)"""
    
    # Program ID
    pid = tl.program_id(0)
    grid_bc = (batch_channels + BLOCK_SIZE_BC - 1) // BLOCK_SIZE_BC
    grid_o = (output_features + BLOCK_SIZE_O - 1) // BLOCK_SIZE_O
    grid_hw = (height * width + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    
    bc = pid % grid_bc
    o = (pid // grid_bc) % grid_o
    hw = (pid // (grid_bc * grid_o)) % grid_hw
    
    # Offsets for the block
    bc_offset = bc * BLOCK_SIZE_BC
    o_offset = o * BLOCK_SIZE_O
    hw_offset = hw * BLOCK_SIZE_HW
    
    # Create masks
    bc_mask = bc_offset + tl.arange(0, BLOCK_SIZE_BC) < batch_channels
    o_mask = o_offset + tl.arange(0, BLOCK_SIZE_O) < output_features
    hw_mask = hw_offset + tl.arange(0, BLOCK_SIZE_HW) < height * width
    i_mask = tl.arange(0, BLOCK_SIZE_I) < input_features
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_BC, BLOCK_SIZE_O, BLOCK_SIZE_HW), dtype=tl.float32)
    
    # Main computation loop
    for i in range(0, input_features, BLOCK_SIZE_I):
        # Load A matrix block: (BC,O,I)
        a_ptrs = a_ptr + (bc_offset[:, None, None] * output_features * input_features +
                          o_offset[None, :, None] * input_features +
                          (i + tl.arange(0, BLOCK_SIZE_I))[None, None, :])
        a_block = tl.load(a_ptrs, mask=bc_mask[:, None, None] & o_mask[None, :, None] & i_mask[None, None, :], other=0.0)
        
        # Load B matrix block: (BC,I,H*W)
        b_ptrs = b_ptr + (bc_offset[:, None, None] * input_features * (height * width) +
                          (i + tl.arange(0, BLOCK_SIZE_I))[:, None, None] * (height * width) +
                          hw_offset[None, None, :] + tl.arange(0, BLOCK_SIZE_HW)[None, None, :])
        b_block = tl.load(b_ptrs, mask=bc_mask[:, None, None] & i_mask[:, None, None] & hw_mask[None, None, :], other=0.0)
        
        # Transform and perform matrix multiplication
        # a_block: (BC,O,I), b_block: (BC,I,HW) -> result: (BC,O,HW)
        a_transposed = a_block  # Already in transpose-like form for efficient computation
        b_reshaped = b_block
        
        # Batch matrix multiplication
        for bc_idx in range(BLOCK_SIZE_BC):
            for o_idx in range(BLOCK_SIZE_O):
                if bc_mask[bc_idx] and o_mask[o_idx]:
                    # Vector dot product for this BC,O position across I dimension
                    for i_idx in range(BLOCK_SIZE_I):
                        if i_mask[i_idx]:
                            a_val = a_transposed[bc_idx, o_idx, i_idx]
                            b_val = b_reshaped[bc_idx, i_idx, :]
                            accumulator[bc_idx, o_idx, :] += a_val * b_val
    
    # Store result: transform from (BC,O,H*W) to output layout
    output_base = bc_offset * output_features * (height * width) + o_offset * (height * width) + hw_offset
    output_ptrs = output_ptr + output_base + tl.arange(0, BLOCK_SIZE_HW)[None, None, :]
    
    # Store the accumulated result
    for bc_idx in range(BLOCK_SIZE_BC):
        for o_idx in range(BLOCK_SIZE_O):
            if bc_mask[bc_idx] and o_mask[o_idx]:
                # Store the HW vector for this BC,O combination
                tl.store(output_ptrs[bc_idx, o_idx, :], accumulator[bc_idx, o_idx, :].to(tl.float16), mask=hw_mask)

@triton.jit
def yolo_matmul_kernel_bf16(
    a_ptr, b_ptr, output_ptr,
    batch_channels, output_features, input_features, height, width,
    BLOCK_SIZE_BC: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    """BF16 optimized version of YOLO kernel"""
    
    pid = tl.program_id(0)
    grid_bc = (batch_channels + BLOCK_SIZE_BC - 1) // BLOCK_SIZE_BC
    grid_o = (output_features + BLOCK_SIZE_O - 1) // BLOCK_SIZE_O
    grid_hw = (height * width + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    
    bc = pid % grid_bc
    o = (pid // grid_bc) % grid_o
    hw = (pid // (grid_bc * grid_o)) % grid_hw
    
    bc_offset = bc * BLOCK_SIZE_BC
    o_offset = o * BLOCK_SIZE_O
    hw_offset = hw * BLOCK_SIZE_HW
    
    bc_mask = bc_offset + tl.arange(0, BLOCK_SIZE_BC) < batch_channels
    o_mask = o_offset + tl.arange(0, BLOCK_SIZE_O) < output_features
    hw_mask = hw_offset + tl.arange(0, BLOCK_SIZE_HW) < height * width
    i_mask = tl.arange(0, BLOCK_SIZE_I) < input_features
    
    accumulator = tl.zeros((BLOCK_SIZE_BC, BLOCK_SIZE_O, BLOCK_SIZE_HW), dtype=tl.bfloat16)
    
    for i in range(0, input_features, BLOCK_SIZE_I):
        a_ptrs = a_ptr + (bc_offset[:, None, None] * output_features * input_features +
                          o_offset[None, :, None] * input_features +
                          (i + tl.arange(0, BLOCK_SIZE_I))[None, None, :])
        a_block = tl.load(a_ptrs, mask=bc_mask[:, None, None] & o_mask[None, :, None] & i_mask[None, None, :], other=0.0)
        
        b_ptrs = b_ptr + (bc_offset[:, None, None] * input_features * (height * width) +
                          (i + tl.arange(0, BLOCK_SIZE_I))[:, None, None] * (height * width) +
                          hw_offset[None, None, :] + tl.arange(0, BLOCK_SIZE_HW)[None, None, :])
        b_block = tl.load(b_ptrs, mask=bc_mask[:, None, None] & i_mask[:, None, None] & hw_mask[None, None, :], other=0.0)
        
        a_transposed = a_block
        b_reshaped = b_block
        
        # Perform BF16 matrix multiplication
        for bc_idx in range(BLOCK_SIZE_BC):
            for o_idx in range(BLOCK_SIZE_O):
                if bc_mask[bc_idx] and o_mask[o_idx]:
                    for i_idx in range(BLOCK_SIZE_I):
                        if i_mask[i_idx]:
                            a_val = a_transposed[bc_idx, o_idx, i_idx]
                            b_val = b_reshaped[bc_idx, i_idx, :]
                            accumulator[bc_idx, o_idx, :] += a_val * b_val
    
    output_base = bc_offset * output_features * (height * width) + o_offset * (height * width) + hw_offset
    output_ptrs = output_ptr + output_base + tl.arange(0, BLOCK_SIZE_HW)[None, None, :]
    
    for bc_idx in range(BLOCK_SIZE_BC):
        for o_idx in range(BLOCK_SIZE_O):
            if bc_mask[bc_idx] and o_mask[o_idx]:
                tl.store(output_ptrs[bc_idx, o_idx, :], accumulator[bc_idx, o_idx, :], mask=hw_mask)

# Optimized kernel wrapper for YOLO-style operations
@torch.fx.wrap
def optimized_yolo_matmul(a, b, target_view_shape):
    """High-performance YOLO-style matmul with view fusion
    
    Args:
        a: [B, C, O, I] - weight-like tensor
        b: [B, C, I, H, W] - input feature maps
        target_view_shape: [B, C*O, H, W] - final view shape after combining C and O
    """
    # Validate input shapes
    assert len(a.shape) == 4, f"Expected 4D tensor for 'a', got {a.shape}"
    assert len(b.shape) == 5, f"Expected 5D tensor for 'b', got {b.shape}"
    assert a.shape[0] == b.shape[0], f"Batch size mismatch: {a.shape[0]} vs {b.shape[0]}"
    assert a.shape[1] == b.shape[1], f"Channel mismatch: {a.shape[1]} vs {b.shape[1]}"
    assert a.shape[-1] == b.shape[-3], f"Input features mismatch: {a.shape[-1]} vs {b.shape[-3]}"
    
    B, C, O, I = a.shape
    _, _, _, H, W = b.shape
    
    # Combine batch and channels for efficient computation
    batch_channels = B * C
    total_output_elements = batch_channels * O * H * W
    
    # Select kernel based on data type
    if a.dtype == torch.bfloat16:
        kernel = yolo_matmul_kernel_bf16
        dtype = torch.bfloat16
    else:
        kernel = yolo_matmul_kernel_fp16
        dtype = torch.float16
    
    # Create output tensor
    output_shape = (batch_channels, O, H * W)
    output = torch.empty(output_shape, dtype=dtype, device=a.device)
    
    # Configure optimal block sizes based on typical YOLO dimensions
    hw_size = H * W
    if hw_size > 400:  # High resolution (e.g., 20x20 = 400)
        block_size_hw = 64
    elif hw_size > 200:
        block_size_hw = 32
    else:
        block_size_hw = 16
    
    if O > 256:
        block_size_o = 32
    elif O > 128:
        block_size_o = 16
    else:
        block_size_o = min(O, 8)
    
    if I > 256:
        block_size_i = 32
    elif I > 128:
        block_size_i = 16
    else:
        block_size_i = min(I, 8)
    
    if batch_channels > 512:
        block_size_bc = 64
    elif batch_channels > 256:
        block_size_bc = 32
    else:
        block_size_bc = 16
    
    # Calculate grid size
    grid_bc = (batch_channels + block_size_bc - 1) // block_size_bc
    grid_o = (O + block_size_o - 1) // block_size_o
    grid_hw = (hw_size + block_size_hw - 1) // block_size_hw
    grid_size = grid_bc * grid_o * grid_hw
    
    # Launch kernel
    kernel[grid_size](
        a, b, output,
        batch_channels, O, I, H, W,
        block_size_bc, block_size_o, block_size_i, block_size_hw
    )
    
    # Apply view operation: [BC,O,H*W] -> [B,C,O,H,W] -> [B,C×O,H,W]
    if target_view_shape is not None and len(target_view_shape) == 4:
        B_out, CO_out, H_out, W_out = target_view_shape
        
        # Reshape to separate batch, channels, and spatial dimensions
        output = output.reshape(B, C, O, H, W)
        
        # Combine channels and output features: [B,C×O,H,W]
        output = output.reshape(B, C * O, H, W)
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    def optimized_func(a, b):
        # Infer target view from YOLO pattern
        B, C, O, _ = a.shape
        _, _, _, H_in, W_in = b.shape
        target_view = (B, C * O, H_in, W_in)
        return optimized_yolo_matmul(a, b, target_view)
    
    return optimized_func