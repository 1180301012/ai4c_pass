import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    # Flatten operation
    tmp_6 = input_tensor.flatten(2)
    # Transpose operation
    tmp_7 = tmp_6.transpose(1, 2)
    return tmp_7

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def flatten_transpose_kernel(
    input_ptr,    # Input tensor [B, C, H, W]
    output_ptr,   # Output tensor [B, H*W, C]
    B, C, H, W,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Each program handles a tile of the output
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # Compute ranges
    hw_total = H * W
    c_offset = pid_c * BLOCK_SIZE_C
    hw_offset = pid_hw * BLOCK_SIZE_HW
    
    # Create masks
    c_mask = c_offset + tl.arange(0, BLOCK_SIZE_C) < C
    hw_mask = hw_offset + tl.arange(0, BLOCK_SIZE_HW) < hw_total
    
    # Process each block efficiently
    for c in range(0, BLOCK_SIZE_C):
        if c_offset + c < C:
            for hw in range(0, BLOCK_SIZE_HW):
                if hw_offset + hw < hw_total:
                    # Compute original 2D coordinates from flattened index
                    h = (hw_offset + hw) // W
                    w = (hw_offset + hw) % W
                    
                    # Compute output index
                    output_idx = ((pid_b * hw_total + (hw_offset + hw)) * C + (c_offset + c))
                    
                    # Load and store with transpose semantics
                    input_idx = ((pid_b * C + (c_offset + c)) * H + h) * W + w
                    val = tl.load(input_ptr + input_idx)
                    tl.store(output_ptr + output_idx, val)

@torch.fx.wrap
def fused_flatten_transpose(input_tensor):
    B, C, H, W = input_tensor.shape
    HW = H * W
    
    output = torch.empty((B, HW, C), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Configure block sizes for better GPU utilization
    BLOCK_SIZE_C = 32
    BLOCK_SIZE_HW = 128
    
    # Calculate grid size
    grid_hw = triton.cdiv(HW, BLOCK_SIZE_HW)
    grid_c = triton.cdiv(C, BLOCK_SIZE_C)
    
    # Launch kernel
    flatten_transpose_kernel[(B, grid_hw, grid_c)](
        input_tensor,
        output,
        B, C, H, W,
        BLOCK_SIZE_C,
        BLOCK_SIZE_HW
    )
    
    return output

def replacement_func():
    return fused_flatten_transpose