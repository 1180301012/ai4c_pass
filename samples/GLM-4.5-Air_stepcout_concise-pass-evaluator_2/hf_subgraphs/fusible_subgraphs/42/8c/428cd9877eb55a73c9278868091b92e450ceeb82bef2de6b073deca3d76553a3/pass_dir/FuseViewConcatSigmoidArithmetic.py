import torch
import triton
import triton.language as tl

def pattern(conv_out, in_3, in_4):
    # Pattern: view + concat + sigmoid + arithmetic operations
    # Always apply view operation to match both patterns
    batch_size = conv_out.shape[0]
    reshaped = conv_out.view(batch_size, 1, -1)
    concatenated = torch.cat([in_3, in_4, reshaped], 2)
    activated = concatenated.sigmoid()
    result = (activated - 0.25) * 3.141592653589793
    return result

def replacement_args(conv_out, in_3, in_4):
    return (conv_out, in_3, in_4)

@triton.jit
def fused_kernel(
    conv_out_ptr, in_3_ptr, in_4_ptr, out_ptr,
    conv_out_size, in_3_size, in_4_size,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Calculate total elements for concatenated tensors
    total_size = conv_out_size + in_3_size + in_4_size
    
    # Each program handles a block of the concatenated output
    block_size = BLOCK_SIZE
    total_blocks = (total_size + block_size - 1) // block_size
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < total_size
    
    # Load input data based on offset position
    if block_start < conv_out_size:
        # Load from conv_out tensor
        load_start = tl.maximum(0, block_start)
        load_end = tl.minimum(block_start + block_size, conv_out_size)
        conv_offset = tl.arange(load_start, load_end) - block_start
        conv_data = tl.load(conv_out_ptr + conv_offset, other=0.0)
    else:
        conv_data = 0.0
    
    if block_start < in_3_size and block_start + block_size > conv_out_size:
        # Load from in_3 tensor
        in_3_start = tl.maximum(conv_out_size, block_start) - conv_out_size
        in_3_end = tl.minimum(conv_out_size + in_3_size, block_start + block_size) - conv_out_size
        if in_3_start < in_3_end:
            in_3_offset = tl.arange(in_3_start, in_3_end) - (tl.maximum(conv_out_size, block_start) - conv_out_size)
            in_3_data = tl.load(in_3_ptr + in_3_offset, other=0.0)
        else:
            in_3_data = 0.0
    else:
        in_3_data = 0.0
    
    if block_start >= conv_out_size + in_3_size:
        # Load from conv_out tensor (after concat perspective)
        conv_offset_start = block_start - (conv_out_size + in_3_size)
        conv_offset_end = block_start + block_size - (conv_out_size + in_3_size)
        if conv_offset_start < conv_out_size and conv_offset_start < conv_offset_end:
            conv_data = tl.load(conv_out_ptr + conv_offset_start, other=0.0)
        else:
            conv_data = 0.0
    
    if block_start + block_size > conv_out_size + in_3_size and block_start < conv_out_size + in_3_size + in_4_size:
        # Load from in_4 tensor
        in_4_start = tl.maximum(0, block_start - conv_out_size - in_3_size)
        in_4_end = tl.minimum(in_4_size, block_start + block_size - conv_out_size - in_3_size)
        if in_4_start < in_4_end:
            in_4_offset = tl.arange(in_4_start, in_4_end)
            in_4_data = tl.load(in_4_ptr + in_4_offset, other=0.0)
        else:
            in_4_data = 0.0
    else:
        in_4_data = 0.0
    
    # Get the actual data for this block
    if block_start < conv_out_size:
        # Conv data region
        end_idx = tl.minimum(conv_out_size, block_start + block_size)
        actual_size = end_idx - block_start
        data = conv_data[:actual_size]
    elif block_start < conv_out_size + in_3_size:
        # in_3 data region
        start_idx = block_start - conv_out_size
        end_idx = tl.minimum(in_3_size, block_start + block_size - conv_out_size)
        actual_size = end_idx - start_idx
        data = in_3_data[:actual_size]
    elif block_start < conv_out_size + in_3_size + in_4_size:
        # in_4 data region
        start_idx = block_start - conv_out_size - in_3_size
        end_idx = tl.minimum(in_4_size, block_start + block_size - conv_out_size - in_3_size)
        actual_size = end_idx - start_idx
        data = in_4_data[:actual_size]
    else:
        data = 0.0
    
    # Apply fused operations: sigmoid + (x - 0.25) * alpha
    sigmoid = 1.0 / (1.0 + tl.exp(-data))
    result = (sigmoid - 0.25) * alpha
    
    # Store the result
    store_offsets = offsets[mask]
    tl.store(out_ptr + store_offsets, result[mask], mask=mask)

@torch.fx.wrap
def fused_operation(conv_out, in_3, in_4):
    # Get tensor sizes
    conv_out_size = conv_out.numel()
    in_3_size = in_3.numel()
    in_4_size = in_4.numel()
    
    # Calculate total output size
    total_size = conv_out_size + in_3_size + in_4_size
    
    # Output tensor
    out = torch.empty(total_size, dtype=torch.float32, device=conv_out.device)
    
    # Triton kernel launch configuration
    BLOCK_SIZE = 1024
    num_programs = (total_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    fused_kernel[(num_programs,)](
        conv_out_ptr=conv_out,
        in_3_ptr=in_3,
        in_4_ptr=in_4,
        out_ptr=out,
        conv_out_size=conv_out_size,
        in_3_size=in_3_size,
        in_4_size=in_4_size,
        alpha=3.141592653589793,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to match original concatenated shape
    # The shape will be determined by input dimensions
    batch_size = conv_out.shape[0]
    return out.view(batch_size, 1, -1)

def replacement_func():
    return fused_operation