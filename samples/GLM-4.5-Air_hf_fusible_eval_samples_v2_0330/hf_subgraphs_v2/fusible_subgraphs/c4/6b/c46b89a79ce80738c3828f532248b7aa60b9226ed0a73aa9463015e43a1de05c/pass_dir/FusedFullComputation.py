import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1, in_2):
    # First unfold chain
    tmp_0 = torch.nn.functional.unfold(in_1, kernel_size=(384, 384), stride=(192, 192))
    tmp_1 = tmp_0.permute(2, 0, 1)
    tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
    
    # Second unfold chain  
    tmp_3 = torch.nn.functional.unfold(in_2, kernel_size=(384, 384), stride=(288, 288))
    tmp_4 = tmp_3.permute(2, 0, 1)
    tmp_5 = tmp_4.reshape(-1, 3, 384, 384)
    
    # Concatenation and type conversion
    tmp_6 = torch.cat([tmp_5, tmp_2, in_0], dim=0)
    tmp_7 = tmp_6.to(dtype=torch.float16)
    
    return tmp_7

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_full_computation_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    output_ptr,
    # in_1 tensor shapes
    in_1_n, in_1_c, in_1_h, in_1_w,
    # in_2 tensor shapes  
    in_2_n, in_2_c, in_2_h, in_2_w,
    # in_0 tensor shape
    in_0_n, in_0_c, in_0_h, in_0_w,
    # Output shape
    output_n, output_c, output_h, output_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= output_n * output_c * output_h * output_w:
        return
    
    # Calculate indices
    idx0 = pid // (output_c * output_h * output_w)
    remainder = pid % (output_c * output_h * output_w)
    idx1 = remainder // (output_h * output_w)
    remainder = remainder % (output_h * output_w)
    idx2 = remainder // output_w
    idx3 = remainder % output_w
    
    data = 0.0
    
    # Determine which input this element comes from and handle the computation
    if idx0 < 9:  # From in_2 patches (9 patches)
        # Calculate patch index
        patch_idx = idx0
        patch_local_idx = patch_idx // 3
        out_x = patch_local_idx % 3
        out_y = patch_local_idx // 3
        
        # Recompute the unfold operation for this patch
        stride_h, stride_w = 288, 288
        kernel_h, kernel_w = 384, 384
        
        patch_y = out_y
        patch_x = out_x
        
        start_y = patch_y * stride_h
        start_x = patch_x * stride_w
        
        # Get channel index within patch
        patch_channel_idx = idx0 % 3
        
        # Load input data
        if (start_y + idx2) < in_2_h and (start_x + idx3) < in_2_w:
            input_idx = ((in_1_n * in_2_n * in_2_c + patch_idx * in_2_c + patch_channel_idx) * 
                        in_2_h * in_2_w + (start_y + idx2) * in_2_w + (start_x + idx3))
            data = tl.load(in_2_ptr + input_idx, other=0.0)
    
    elif idx0 < 18:  # From in_1 patches (9 patches) 
        # Similar computation for in_1 with stride 192
        local_idx = idx0 - 9
        patch_idx = local_idx
        patch_local_idx = patch_idx // 3
        out_x = patch_local_idx % 3
        out_y = patch_local_idx // 3
        
        stride_h, stride_w = 192, 192
        kernel_h, kernel_w = 384, 384
        
        patch_y = out_y
        patch_x = out_x
        
        start_y = patch_y * stride_h
        start_x = patch_x * stride_w
        
        patch_channel_idx = local_idx % 3
        
        if (start_y + idx2) < in_1_h and (start_x + idx3) < in_1_w:
            input_idx = ((in_1_n * in_2_n * in_1_c + patch_idx * in_1_c + patch_channel_idx) * 
                        in_1_h * in_1_w + (start_y + idx2) * in_1_w + (start_x + idx3))
            data = tl.load(in_1_ptr + input_idx, other=0.0)
    
    else:  # From in_0 direct tensor
        local_idx = idx0 - 18
        if local_idx < in_0_n:
            input_idx = (local_idx * in_0_c + idx1) * in_0_h * in_0_w + idx2 * in_0_w + idx3
            data = tl.load(in_0_ptr + input_idx, other=0.0)
    
    # Convert to float16
    output_value = data.to(tl.float16)
    tl.store(output_ptr + pid, output_value)

@torch.fx.wrap
def fused_full_computation(in_0, in_1, in_2):
    # Get input shapes
    in_1_shape = in_1.shape
    in_2_shape = in_2.shape
    in_0_shape = in_0.shape
    
    # Calculate output shape: [9+9+1, 3, 384, 384] = [19, 3, 384, 384]
    output_shape = (19, 3, 384, 384)
    output_elements = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=torch.float16, device=in_0.device)
    
    BLOCK_SIZE = 128
    grid = (output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_full_computation_kernel[grid](
        in_0,
        in_1, 
        in_2,
        output,
        in_1_shape[0], in_1_shape[1], in_1_shape[2], in_1_shape[3],
        in_2_shape[0], in_2_shape[1], in_2_shape[2], in_2_shape[3],
        in_0_shape[0], in_0_shape[1], in_0_shape[2], in_0_shape[3],
        output_shape[0], output_shape[1], output_shape[2], output_shape[3],
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_full_computation