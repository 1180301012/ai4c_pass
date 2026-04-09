import torch
import triton
import triton.language as tl

def pattern(conv_weight, input_tensor):
    # Conv2D with 1x1 kernel, stride 1, padding 0, dilation 1, groups 1
    conv2d = torch.conv2d(input_tensor, conv_weight, None, (1, 1), (0, 0), (1, 1), 1)
    # Unfold with 2x2 kernel, stride 2
    unfold = torch.nn.functional.unfold(conv2d, kernel_size=(2, 2), stride=(2, 2))
    return conv2d, unfold

def replacement_args(conv_weight, input_tensor):
    return (conv_weight, input_tensor)

@triton.jit
def fused_conv_unfold_kernel(
    input_ptr,  # [N, C_in, H_in, W_in] -> [1, 256, 32, 32]
    weight_ptr,  # [C_out, C_in, 1, 1] -> [128, 256, 1, 1]
    output_ptr,  # [N, C_out * 4, H_out, W_out] -> [1, 512, 16, 16] 
    N, C_in, H_in, W_in, C_out,
    kernel_h: tl.constexpr, kernel_w: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Each program handles one output spatial position
    pid = tl.program_id(axis=0)
    grid_h = (H_in - kernel_h) // stride_h + 1
    grid_w = (W_in - kernel_w) // stride_w + 1
    h_out = pid // grid_w
    w_out = pid % grid_w
    
    # Only process within bounds
    if h_out < grid_h and w_out < grid_w:
        # For each output channel block
        for c_out_block in range(0, C_out, BLOCK_SIZE_M):
            # Process each of the 4 output features (2x2 patch)
            for patch_idx in range(4):
                patch_h = patch_idx // 2  # 0 or 1
                patch_w = patch_idx % 2   # 0 or 1
                
                # Calculate input position for this patch element
                ih = h_out * stride_h + patch_h
                iw = w_out * stride_w + patch_w
                
                # Only process if within bounds
                if ih < H_in and iw < W_in:
                    # Sum over input channels for this patch position
                    sum_val = 0.0
                    for c_in in range(C_in):
                        # Load input
                        input_offset = N * C_in * H_in * W_in + c_in * H_in * W_in + ih * W_in + iw
                        input_val = tl.load(input_ptr + input_offset)
                        
                        # Load weight (use c_out_block as base)
                        weight_offset = (c_out_block + (patch_idx // 2)) * C_in + c_in
                        weight_val = tl.load(weight_ptr + weight_offset)
                        
                        sum_val += input_val * weight_val
                    
                    # Store output at correct position
                    # Output layout: [batch, output_channel, patch_idx, spatial_position]
                    spatial_idx = h_out * grid_w + w_out
                    output_offset = N * C_out * 4 * spatial_idx + c_out_block * 4 + patch_idx
                    tl.store(output_ptr + output_offset, sum_val)

@torch.fx.wrap
def fused_conv_unfold(conv_weight, input_tensor):
    N, C_in, H_in, W_in = input_tensor.shape
    C_out, _, kernel_h, kernel_w = conv_weight.shape
    
    # For this specific case: unfold output is [N, C_out * 4, H_out, W_out]
    H_out = (H_in - kernel_h) // 2 + 1
    W_out = (W_in - kernel_w) // 2 + 1
    output_channels = C_out * 4
    
    output = torch.zeros([N, output_channels, H_out, W_out], dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set up grid: one thread per spatial position 
    grid_size = H_out * W_out
    if grid_size > 0:
        fused_conv_unfold_kernel[(grid_size,)](
            input_ptr=input_tensor,
            weight_ptr=conv_weight,  # Use original 1x1 weight
            output_ptr=output,
            N=N, C_in=C_in, H_in=H_in, W_in=W_in, C_out=C_out,
            kernel_h=kernel_h, kernel_w=kernel_w,
            stride_h=2, stride_w=2,
            BLOCK_SIZE_M=32,
        )
    
    return output

def replacement_func():
    return fused_conv_unfold