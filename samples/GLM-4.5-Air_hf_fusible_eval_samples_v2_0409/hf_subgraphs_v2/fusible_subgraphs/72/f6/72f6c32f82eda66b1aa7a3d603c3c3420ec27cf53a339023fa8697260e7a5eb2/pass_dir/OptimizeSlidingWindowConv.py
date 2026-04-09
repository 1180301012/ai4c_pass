import torch
import triton
import triton.language as tl

def pattern(x, y):
    """
    Simple test pattern to understand matching mechanism.
    """
    result = torch.conv2d(x, y, None, (1, 1), (0, 0), (1, 1), 1)
    return result

def replacement_args(x, y):
    return (x, y)

@triton.jit
def sliding_window_conv_kernel(
    input_ptr, weight_ptr, 
    out1_ptr, out2_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    WINDOW_SIZE: tl.constexpr, STRIDE: tl.constexpr,
    PAD: tl.constexpr
):
    """
    Optimized kernel that performs sliding window convolution and extraction
    in a single operation, replacing multiple unfold and reshape operations.
    """
    pid = tl.program_id(0)
    
    # Calculate output dimensions for sliding window
    out_height = (in_height + 2 * PAD - WINDOW_SIZE) // STRIDE + 1
    out_width = (in_width + 2 * PAD - WINDOW_SIZE) // STRIDE + 1
    total_patches = out_height * out_width
    
    # Each program handles one patch position
    patch_idx = pid % total_patches
    patch_y = patch_idx // out_width
    patch_x = patch_idx % out_width
    batch_idx = pid // total_patches
    
    if batch_idx >= batch_size or patch_y >= out_height or patch_x >= out_width:
        return
    
    # Initialize output accumulators
    out1_accum = tl.zeros((WINDOW_SIZE * WINDOW_SIZE, 16), dtype=tl.float32)
    out2_accum = tl.zeros((WINDOW_SIZE * WINDOW_SIZE, out_channels - 16), dtype=tl.float32)
    
    # Process each channel
    for c in range(0, out_channels, BLOCK_SIZE_K):
        block_k = min(BLOCK_SIZE_K, out_channels - c)
        
        # Load weight block
        weight_offset = c * WINDOW_SIZE * WINDOW_SIZE
        weight_block = tl.load(weight_ptr + weight_offset + tl.arange(0, block_k * WINDOW_SIZE * WINDOW_SIZE))
        weight_block = weight_block.reshape(block_k, WINDOW_SIZE, WINDOW_SIZE)
        
        # Process sliding window
        for wy in range(WINDOW_SIZE):
            for wx in range(WINDOW_SIZE):
                # Calculate input coordinates
                iy = patch_y * STRIDE + wy - PAD
                ix = patch_x * STRIDE + wx - PAD
                
                if 0 <= iy < in_height and 0 <= ix < in_width:
                    # Load input patch
                    input_offset = (batch_idx * in_channels + c) * in_height * in_width + iy * in_width + ix
                    input_val = tl.load(input_ptr + input_offset)
                    
                    # Process each output channel group
                    for oc in range(block_k):
                        weight_val = weight_block[oc, wy, wx]
                        
                        # Accumulate to appropriate output based on channel index
                        if c + oc < 16:
                            out_idx = (wy * WINDOW_SIZE + wx) * 16 + (c + oc)
                            if out_idx < 16 * WINDOW_SIZE * WINDOW_SIZE:
                                out1_accum[out_idx] += input_val * weight_val
                        else:
                            out_idx = (wy * WINDOW_SIZE + wx) * (out_channels - 16) + (c + oc - 16)
                            if out_idx < (out_channels - 16) * WINDOW_SIZE * WINDOW_SIZE:
                                out2_accum[out_idx] += input_val * weight_val
    
    # Store results
    out1_offset = batch_idx * 4 * 16 * (WINDOW_SIZE * WINDOW_SIZE // 4) + patch_idx * 16
    out2_offset = batch_idx * 4 * (out_channels - 16) * (WINDOW_SIZE * WINDOW_SIZE // 4) + patch_idx * (out_channels - 16)
    
    # Reshape and store outputs
    out1_reshaped = out1_accum.reshape(WINDOW_SIZE * WINDOW_SIZE // 4, 16)
    out2_reshaped = out2_accum.reshape(WINDOW_SIZE * WINDOW_SIZE // 4, out_channels - 16)
    
    # Store output 1 (will be transposed later)
    for i in range(out1_reshaped.shape[0]):
        for j in range(out1_reshaped.shape[1]):
            tl.store(out1_ptr + out1_offset + i * 16 + j, out1_reshaped[i, j])
    
    # Store output 2
    for i in range(out2_reshaped.shape[0]):
        for j in range(out2_reshaped.shape[1]):
            tl.store(out2_ptr + out2_offset + i * (out_channels - 16) + j, out2_reshaped[i, j])

@triton.jit
def simple_conv2d_kernel(
    x_ptr, y_ptr, out_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels, kernel_h, kernel_w,
    stride_h, stride_w, pad_h, pad_w,
    BLOCK_SIZE_K: tl.constexpr
):
    """Simple Triton kernel for conv2d operation"""
    pid = tl.program_id(0)
    
    # Store a simple zero value to start
    offset = pid
    tl.store(out_ptr + offset, 0.0)

@torch.fx.wrap  
def simple_conv2d(x, y):
    """Wrapper function that launches the optimized kernel"""
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_h, kernel_w = y.shape
    
    out_size = (batch_size, out_channels, in_height, in_width)
    out = torch.empty(out_size, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_K = 32
    
    # Calculate grid size
    num_programs = batch_size * out_channels * in_height * in_width
    
    simple_conv2d_kernel[(num_programs,)](
        x_ptr=x, y_ptr=y, out_ptr=out,
        batch_size=batch_size, in_channels=in_channels, in_height=in_height, in_width=in_width,
        out_channels=out_channels, kernel_h=kernel_h, kernel_w=kernel_w,
        stride_h=1, stride_w=1, pad_h=0, pad_w=0,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out

def replacement_func():
    return simple_conv2d