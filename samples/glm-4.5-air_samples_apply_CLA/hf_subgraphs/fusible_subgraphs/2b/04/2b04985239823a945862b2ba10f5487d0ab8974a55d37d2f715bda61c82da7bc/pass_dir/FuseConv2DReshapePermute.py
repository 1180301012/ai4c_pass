import torch
import triton
import triton.language as tl

def pattern(tmp_4, B, C):
    """
    Pattern matching for Conv2D + Reshape + Permute fusion
    tmp_4: output of conv2d operation [B, C, H_out, W_out]
    B: batch size
    C: number of channels
    """
    # Reshape operation: reshape to [B, C, -1] 
    tmp_5 = tmp_4.reshape(B, C, -1)
    
    # Permute operation: transpose to [B, H*W, C]
    tmp_6 = tmp_5.permute(0, 2, 1)
    
    return tmp_4, tmp_5, tmp_6

def replacement_args(in_4, tmp_3, tmp_2):
    """Extract arguments for fused convolution + reshape + permute kernel"""
    B, C, H, W = in_4.shape
    
    return (in_4, tmp_3, tmp_2, B, C, H, W)

@triton.jit
def fused_reshape_permute_kernel(
    input_ptr,  # [B, C, H, W]
    output_ptr,  # [B, H*W, C]
    C: tl.constexpr,
    HW_total: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused reshape + permute kernel
    Reshapes from [B, C, H, W] to [B, H*W, C] in a single kernel"""
    pid_m = tl.program_id(0)  # Batch position
    pid_n = tl.program_id(1)  # Spatial position
    
    # Output offset for current batch and spatial position
    output_base = pid_m * C * HW_total * 4 + pid_n * C * 4
    spatial_offset = pid_n
    
    # Process channels in blocks
    channel_offset = tl.arange(0, BLOCK_SIZE_M)
    channel_mask = channel_offset < C
    
    # Load input data: [B, C, H, W] -> [B, C, H*W]
    input_offset = pid_m * C * HW_total * 4 + channel_offset * HW_total * 4 + spatial_offset * 4
    input_data = tl.load(input_ptr + input_offset, mask=channel_mask, other=0.0)
    
    # Store output data: [B, H*W, C]
    output_offset = output_base + channel_offset * 4
    tl.store(output_ptr + output_offset, input_data, mask=channel_mask)

@torch.fx.wrap  
def fused_conv_reshape_permute(in_4, tmp_3, tmp_2, B, C, H, W):
    """Fused convolution + reshape + permute kernel wrapper"""
    # Assume stride=(1,1), padding=(0,0), dilation=(1,1), groups=1 as per the model
    stride = (1, 1)
    padding = (0, 0)
    dilation = (1, 1)
    groups = 1
    
    # Determine optimal block sizes
    C_total = C
    HW_total = H * W
    
    # Heuristic for block sizes based on tensor dimensions
    if C_total <= 64:
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        GROUP_SIZE_M = 1
    elif C_total <= 320:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
        GROUP_SIZE_M = 4
    else:
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 256
        BLOCK_SIZE_K = 64
        GROUP_SIZE_M = 4
    
    # Calculate grid dimensions
    num_programs_m = tl.cdiv(C_total, BLOCK_SIZE_M * GROUP_SIZE_M)
    num_programs_n = tl.cdiv(HW_total, BLOCK_SIZE_N)
    
    # Create output tensor in transposed format [B, H*W, C]
    output_shape = (B, HW_total, C_total)
    output = torch.empty(output_shape, dtype=torch.float32, device=in_4.device)
    
    # Perform regular convolution first (since we're only fusing reshape + permute)
    conv_output = torch.conv2d(in_4, tmp_3, tmp_2, stride, padding, dilation, groups)
    
    # Now fuse reshape + permute
    fused_reshape_permute_kernel[(num_programs_m, num_programs_n)](
        conv_output,
        output,
        C_total, HW_total,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    return fused_conv_reshape_permute