import torch
import triton
import triton.language as tl

def pattern(in_3, in_1, in_0):
    """Match Conv2D + View pattern - just the conv2d and view operations"""
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(conv2d.shape[0], 256, -1)
    return tmp_3

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@triton.jit
def fused_conv2d_view_kernel(
    input_ptr,      # [N, 512, 64, 64]
    weight_ptr,     # [256, 512, 1, 1] 
    bias_ptr,       # [256]
    output_ptr,     # [N, 256, 4096]
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output channel
    c = tl.program_id(0)
    
    # Load bias for this channel
    bias_val = tl.load(bias_ptr + c)
    
    # Calculate output coordinates
    n = tl.program_id(1)  # batch dimension
    h_offset = tl.program_id(2) * 64  # 64 spatial locations per program
    w_offset = 0
    
    # Process all 4096 elements for this (n, c) pair
    for h in range(64):
        for w in range(64):
            # Compute input offset
            in_offset = n * 512 * 64 * 64 + h_offset + h * 64 + w
            
            # Load 512 input channels for this spatial location
            input_vals = tl.load(input_ptr + in_offset, mask=None)
            
            # Compute weighted sum
            weighted_sum = tl.sum(input_vals * tl.load(weight_ptr + c * 512))
            
            # Store result with bias
            out_offset = n * 256 * 4096 + c * 4096 + (h_offset + h) * 64 + w
            tl.store(output_ptr + out_offset, weighted_sum + bias_val)

@torch.fx.wrap
def fused_conv2d_view(in_3, in_1, in_0):
    """Fused Conv2D + View operation"""
    N, C_in, H, W = in_3.shape
    C_out = in_0.shape[0]
    
    output = torch.empty((N, C_out, H * W), dtype=in_3.dtype, device=in_3.device)
    
    # Launch kernel for conv2d fused with view
    fused_conv2d_view_kernel[(C_out, N, (W + 63) // 64)](
        in_3,
        in_1, 
        in_0,
        output,
        N,
        BLOCK_SIZE=1,
    )
    
    return output

def replacement_func():
    return fused_conv2d_view