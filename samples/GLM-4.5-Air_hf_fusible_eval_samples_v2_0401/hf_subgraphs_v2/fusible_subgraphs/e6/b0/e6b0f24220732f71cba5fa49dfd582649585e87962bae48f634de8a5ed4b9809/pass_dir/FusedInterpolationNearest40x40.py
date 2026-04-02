import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Pattern to match two separate interpolations to the same target size"""
    tmp_1 = torch.nn.functional.interpolate(in_0, size=(40, 40), mode='nearest')
    tmp_2 = torch.nn.functional.interpolate(in_1, size=(40, 40), mode='nearest')
    return tmp_1, tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_interpolation_kernel(
    input0_ptr,
    input1_ptr,
    output0_ptr, 
    output1_ptr,
    batch_size,
    channels0,
    channels1,
    orig_height0,
    orig_width0,
    orig_height1,
    orig_width1,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused nearest neighbor interpolation kernel for two inputs"""
    pid = tl.program_id(0)
    # Process each batch element in parallel
    batch_idx = pid
    
    if batch_idx >= batch_size:
        return
        
    # Calculate offsets for each batch element
    input0_base = batch_idx * channels0 * orig_height0 * orig_width0
    input1_base = batch_idx * channels1 * orig_height1 * orig_width1
    output0_base = batch_idx * channels0 * 40 * 40
    output1_base = batch_idx * channels1 * 40 * 40
    
    # Process each channel and output pixel
    for c0 in range(channels0):
        for h in range(40):
            for w in range(40):
                # Calculate source coordinates (nearest neighbor)
                src_h0 = h * orig_height0 // 40
                src_w0 = w * orig_width0 // 40
                src_idx0 = input0_base + c0 * orig_height0 * orig_width0 + src_h0 * orig_width0 + src_w0
                
                # Load and store output 0
                dst_idx0 = output0_base + c0 * 40 * 40 + h * 40 + w
                val0 = tl.load(input0_ptr + src_idx0)
                tl.store(output0_ptr + dst_idx0, val0)
    
    for c1 in range(channels1):
        for h in range(40):
            for w in range(40):
                # Calculate source coordinates (nearest neighbor)
                src_h1 = h * orig_height1 // 40
                src_w1 = w * orig_width1 // 40
                src_idx1 = input1_base + c1 * orig_height1 * orig_width1 + src_h1 * orig_width1 + src_w1
                
                # Load and store output 1
                dst_idx1 = output1_base + c1 * 40 * 40 + h * 40 + w
                val1 = tl.load(input1_ptr + src_idx1)
                tl.store(output1_ptr + dst_idx1, val1)

@torch.fx.wrap
def fused_interpolation(in_0, in_1):
    """Fused interpolation function that processes both inputs in one kernel launch"""
    batch0, channels0, height0, width0 = in_0.shape
    batch1, channels1, height1, width1 = in_1.shape
    
    # Output tensors
    out_0 = torch.empty((batch0, channels0, 40, 40), dtype=in_0.dtype, device=in_0.device)
    out_1 = torch.empty((batch1, channels1, 40, 40), dtype=in_1.dtype, device=in_1.device)
    
    # Calculate grid size
    total_batch_elements = max(batch0, batch1)
    BLOCK_SIZE = 1  # Each program handles one batch element
    grid_size = (total_batch_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch fused kernel
    fused_interpolation_kernel[grid_size](
        in_0,
        in_1,
        out_0,
        out_1,
        total_batch_elements,
        channels0, channels1,
        height0, width0,
        height1, width1,
        BLOCK_SIZE
    )
    
    return out_0, out_1

def replacement_func():
    return fused_interpolation