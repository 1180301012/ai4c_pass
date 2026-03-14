import torch
import triton
import triton.language as tl

@triton.jit
def fused_forward_kernel_768(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, in_4_ptr, in_5_ptr,
    out_ptr,
    feat_height, feat_width, feat_channels,
    batch_size, num_inputs,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Complete fused kernel for 768 channel case"""
    # Program ID for batch and spatial dimensions
    batch_idx = tl.program_id(0)
    h_idx = tl.program_id(1) 
    w_idx = tl.program_id(2)
    
    # Calculate spatial position within this block
    m = tl.arange(0, BLOCK_SIZE_M)
    n = tl.arange(0, BLOCK_SIZE_N)
    mask_m = h_idx * BLOCK_SIZE_M + m < feat_height
    mask_n = w_idx * BLOCK_SIZE_N + n < feat_width
    
    # Flatten spatial indices
    spatial_indices = (h_idx * BLOCK_SIZE_M + m) * feat_width + (w_idx * BLOCK_SIZE_N + n)
    valid_mask = mask_m & mask_n
    
    # Process each spatial position
    for i in range(BLOCK_SIZE_M):
        for j in range(BLOCK_SIZE_N):
            if tl.static_cast<int>(h_idx * BLOCK_SIZE_M + i) < feat_height and \
               tl.static_cast<int>(w_idx * BLOCK_SIZE_N + j) < feat_width:
                
                # Spatial position offset
                pos_idx = (h_idx * BLOCK_SIZE_M + i) * feat_width + (w_idx * BLOCK_SIZE_N + j)
                
                # Load bias and weights for this position
                bias = tl.load(in_0_ptr + tl.arange(0, hidden_size), mask=tl.arange(0, hidden_size) < hidden_size).to(tl.float32)
                weight = tl.load(in_1_ptr + tl.arange(0, hidden_size), mask=tl.arange(0, hidden_size) < hidden_size).to(tl.float32)
                
                # Concatenate and process input features from all 4 inputs
                x_channels = tl.zeros([hidden_size], dtype=tl.float32)
                for input_idx in range(num_inputs):
                    feat_channels_per_input = hidden_size // num_inputs
                    for c in range(feat_channels_per_input):
                        channel_offset = input_idx * feat_channels_per_input + c
                        
                        # Calculate input offset for this spatial position and input
                        input_offset = input_idx * feat_channels * feat_height * feat_width + \
                                       (h_idx * BLOCK_SIZE_M + i) * feat_width * feat_channels + \
                                       (w_idx * BLOCK_SIZE_N + j) * feat_channels + c
                        
                        # Load feature and add to combined channels
                        feat_val = tl.load(in_2_ptr + input_offset + input_idx * feat_channels * feat_height * feat_width, 
                                         other=0.0).to(tl.float32)
                        x_channels[channel_offset] = feat_val
                
                # Compute LayerNorm
                mean = tl.sum(x_channels) / hidden_size
                var = tl.sum((x_channels - mean) * (x_channels - mean)) / hidden_size
                x_norm = (x_channels - mean) * tl.rsqrt(var + eps)
                out = x_norm * weight + bias
                
                # Store result
                output_offset = batch_idx * (feat_height * feat_width * hidden_size) + pos_idx * hidden_size
                tl.store(out_ptr + output_offset + tl.arange(0, hidden_size), out, mask=tl.arange(0, hidden_size) < hidden_size)

@torch.fx.wrap
def fused_forward_768(in_0, in_1, in_2, in_3, in_4, in_5):
    """Complete fused forward function for 768 channel case"""
    # Get input dimensions
    batch_size = 1
    height, width, channels = in_2.shape
    
    # Create output tensor
    output_shape = (batch_size, height * width, 768)
    output = torch.empty(output_shape, dtype=torch.float32, device=in_2.device)
    
    # Launch kernel
    ptrs = [in_2.data_ptr(), in_3.data_ptr(), in_4.data_ptr(), in_5.data_ptr()]
    
    grid = (batch_size, (height + 31) // 32, (width + 31) // 32)
    fused_forward_kernel_768[grid](
        in_0.data_ptr(), in_1.data_ptr(),
        *ptrs,
        output.data_ptr(),
        height, width, channels,
        batch_size, 4,  # 4 input tensors
        768, 1e-05,
        32, 32
    )
    
    return (output,)

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """Pattern: Complete forward computation with concat-view and layer norm"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.cat([in_2, in_3, in_4, in_5], -1)
    tmp_3 = tmp_2.view(1, -1, 768)
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), tmp_1, tmp_0, 1e-05)
    return tmp_4

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    """Extract arguments for replacement"""
    return (in_0, in_1, in_2, in_3, in_4, in_5)

def replacement_func():
    """Return the complete fused forward function"""
    return fused_forward_768