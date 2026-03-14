import torch
import triton
import triton.language as tl

@triton.jit
def concat_view_kernel_768(
    ptrs,
    out_ptr,
    feat_height,
    feat_width,
    feat_channels,
    batch_size,
    num_inputs,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Kernel for fused concatenation and view operation for 768 channel case"""
    # Program ID for batch and spatial dimensions
    batch_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    feat_idx = tl.program_id(2)
    
    # Offset for batch and spatial position
    batch_offset = batch_idx * feat_height * feat_width * 768
    spatial_offset = h_idx * feat_width * 768 + feat_idx * BLOCK_SIZE_N
    
    # Output offset
    out_offset = batch_offset + spatial_offset
    
    # Process each input tensor
    for input_idx in range(num_inputs):
        input_base = input_idx * feat_channels * feat_height * feat_width
        
        # Load input data and write directly to output
        for m in range(BLOCK_SIZE_M):
            for n in range(BLOCK_SIZE_N):
                h_pos = h_idx * BLOCK_SIZE_M + m
                w_pos = feat_idx * BLOCK_SIZE_N + n
                
                if h_pos < feat_height and n < feat_width:
                    input_offset = input_base + h_pos * feat_width * feat_channels + n * feat_channels
                    output_pos = out_offset + m * feat_width * 768 + n
                    
                    # Copy channels directly (192 channels per input, 4 inputs = 768 total)
                    for c in range(192):
                        input_ptr = ptrs[input_idx]
                        tl.store(out_ptr + output_pos * 768 + c + input_idx * 192, 
                                tl.load(input_ptr + input_offset + c, other=0.0))

@torch.fx.wrap
def fused_concat_view_768(in_2, in_3, in_4, in_5):
    """Fused concatenation and view operation for 768 channel case"""
    # Input shapes: [1, 32, 32, 192] each
    batch_size, height, width, channels = in_2.shape
    total_channels = 768
    
    # Create output tensor: [1, 1024, 768]
    output_shape = (batch_size, height * width, total_channels)
    output = torch.empty(output_shape, dtype=torch.float32, device=in_2.device)
    
    # Set up grid dimensions
    num_batches = 1
    num_heights = (height + 31) // 32  # Block size 32 for height
    num_widths = (width + 31) // 32    # Block size 32 for width
    
    # Launch kernel
    ptrs = [in_2.data_ptr(), in_3.data_ptr(), in_4.data_ptr(), in_5.data_ptr()]
    
    grid = (num_batches, num_heights, num_widths)
    concat_view_kernel_768[grid](
        ptrs,
        output.data_ptr(),
        height,
        width,
        channels,
        batch_size,
        4,  # 4 input tensors
        32,  # BLOCK_SIZE_M
        32   # BLOCK_SIZE_N
    )
    
    return output

@triton.jit
def concat_view_kernel_1536(
    ptrs,
    out_ptr,
    feat_height,
    feat_width,
    feat_channels,
    batch_size,
    num_inputs,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Kernel for fused concatenation and view operation for 1536 channel case"""
    # Program ID for batch and spatial dimensions
    batch_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    feat_idx = tl.program_id(2)
    
    # Offset for batch and spatial position
    batch_offset = batch_idx * feat_height * feat_width * 1536
    spatial_offset = h_idx * feat_width * 1536 + feat_idx * BLOCK_SIZE_N
    
    # Output offset
    out_offset = batch_offset + spatial_offset
    
    # Process each input tensor
    for input_idx in range(num_inputs):
        input_base = input_idx * feat_channels * feat_height * feat_width
        
        # Load input data and write directly to output
        for m in range(BLOCK_SIZE_M):
            for n in range(BLOCK_SIZE_N):
                h_pos = h_idx * BLOCK_SIZE_M + m
                w_pos = feat_idx * BLOCK_SIZE_N + n
                
                if h_pos < feat_height and n < feat_width:
                    input_offset = input_base + h_pos * feat_width * feat_channels + n * feat_channels
                    output_pos = out_offset + m * feat_width * 1536 + n
                    
                    # Copy channels directly (384 channels per input, 4 inputs = 1536 total)
                    for c in range(384):
                        input_ptr = ptrs[input_idx]
                        tl.store(out_ptr + output_pos * 1536 + c + input_idx * 384, 
                                tl.load(input_ptr + input_offset + c, other=0.0))

@torch.fx.wrap
def fused_concat_view_1536(in_2, in_3, in_4, in_5):
    """Fused concatenation and view operation for 1536 channel case"""
    # Input shapes: [1, 16, 16, 384] each
    batch_size, height, width, channels = in_2.shape
    total_channels = 1536
    
    # Create output tensor: [1, 256, 1536]
    output_shape = (batch_size, height * width, total_channels)
    output = torch.empty(output_shape, dtype=torch.float32, device=in_2.device)
    
    # Set up grid dimensions
    num_batches = 1
    num_heights = (height + 31) // 32  # Block size 32 for height
    num_widths = (width + 31) // 32    # Block size 32 for width
    
    # Launch kernel
    ptrs = [in_2.data_ptr(), in_3.data_ptr(), in_4.data_ptr(), in_5.data_ptr()]
    
    grid = (num_batches, num_heights, num_widths)
    concat_view_kernel_1536[grid](
        ptrs,
        output.data_ptr(),
        height,
        width,
        channels,
        batch_size,
        4,  # 4 input tensors
        32,  # BLOCK_SIZE_M
        32   # BLOCK_SIZE_N
    )
    
    return output

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """Pattern: tmp_2 = torch.cat([in_2, in_3, in_4, in_5], -1); tmp_3 = tmp_2.view(1, -1, channels)"""
    tmp_2 = torch.cat([in_2, in_3, in_4, in_5], -1)
    # Determine channel size based on input shapes
    if in_2.shape[-1] == 192:  # 4 * 192 = 768 channels
        tmp_3 = tmp_2.view(1, -1, 768)
    elif in_2.shape[-1] == 384:  # 4 * 384 = 1536 channels
        tmp_3 = tmp_2.view(1, -1, 1536)
    else:
        raise ValueError(f"Unsupported input channel size: {in_2.shape[-1]}")
    return tmp_3

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    """Extract arguments for replacement"""
    return (in_2, in_3, in_4, in_5)

def replacement_func():
    """Return a function that selects appropriate fused concatenation based on input channel size"""
    def select_fused_function(in_2, in_3, in_4, in_5):
        if in_2.shape[-1] == 192:  # 4 * 192 = 768 channels
            return fused_concat_view_768(in_2, in_3, in_4, in_5)
        elif in_2.shape[-1] == 384:  # 4 * 384 = 1536 channels
            return fused_concat_view_1536(in_2, in_3, in_4, in_5)
        else:
            raise ValueError(f"Unsupported input channel size: {in_2.shape[-1]}")
    
    return select_fused_function