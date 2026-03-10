import torch
import triton
import triton.language as tl

# Pattern matching function: Conv2D followed by unfold
def pattern(x, weight):
    # Conv2D: input [1, 256, 32, 32] with weight [128, 256, 1, 1] -> [1, 128, 32, 32]
    tmp_1 = torch.conv2d(x, weight, None, (1, 1), (0, 0), (1, 1), 1)
    # Unfold: [1, 128, 32, 32] with kernel_size=(2, 2), stride=(2, 2) -> [1, 512, 256]
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=(2, 2), stride=(2, 2))
    return tmp_2

# Argument extraction function
def replacement_args(x, weight):
    return (x, weight)

# Optimized kernel that fuses Conv2D and Unfold operations
@triton.jit
def fused_conv_unfold_kernel(
    x_ptr,          # Input tensor [1, 256, 32, 32] 
    weight_ptr,     # Weight tensor [128, 256, 1, 1]  
    out_ptr,        # Output tensor [1, 512, 256] = [1, 4*128, 16*16]
    n_channels_in,  # 256
    n_channels_out, # 128
    feat_h,         # 32
    feat_w,         # 32
    unfold_h,       # 16 (output height after unfold with stride 2)
    unfold_w,       # 16 (output width after unfold with stride 2)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    block_start = pid * block_size
    block_end = min(block_start + block_size, n_channels_out * unfold_h * unfold_w)
    
    # Process output elements in this block
    offsets = block_start + tl.arange(0, block_end - block_start)
    mask = offsets < n_channels_out * unfold_h * unfold_w
    
    for idx in offsets:
        if idx >= n_channels_out * unfold_h * unfold_w:
            continue
            
        # Decode output index: [out_c, h_out, w_out]
        out_c = idx // (unfold_h * unfold_w)
        pos_idx = idx % (unfold_h * unfold_w)
        h_out = pos_idx // unfold_w  # 0-15
        w_out = pos_idx % unfold_w   # 0-15
        
        # Map back to input 2x2 patch center
        h_in = h_out * 2  # 0, 2, 4, ..., 30
        w_in = w_out * 2  # 0, 2, 4, ..., 30
        
        # Perform 1x1 convolution using weights for output channel out_c
        conv_result = 0.0
        for c_in in range(n_channels_in):
            # Weight for (out_c, c_in)
            weight_val = tl.load(weight_ptr + out_c * n_channels_in + c_in, mask=True).to(tl.float32)
            
            # We need the value at all 4 locations in the 2x2 patch
            # Since this is 1x1 convolution with same weights everywhere,
            # we just use one location and replicate
            in_offset = c_in * feat_h * feat_w + h_in * feat_w + w_in
            in_val = tl.load(x_ptr + in_offset, mask=True).to(tl.float32)
            
            conv_result += in_val * weight_val
        
        # Store 4 copies of the convolution result in the unfolded format
        # Each unfolded output location corresponds to a 2x2 patch
        for patch_local_idx in range(4):
            # Calculate location within 2x2 patch
            patch_h_local = patch_local_idx // 2  # 0 or 1
            patch_w_local = patch_local_idx % 2   # 0 or 1
            
            # Apply to each position in the 2x2 patch
            patch_h = h_in + patch_h_local
            patch_w = w_in + patch_w_local
            
            if patch_h < feat_h and patch_w < feat_w:
                # Output index: [batch=0, 4*out_c + patch_local_idx, h_out*unfold_w + w_out]
                out_idx = (4 * out_c + patch_local_idx) * unfold_h * unfold_w + h_out * unfold_w + w_out
                tl.store(out_ptr + out_idx, conv_result, mask=True)

@torch.fx.wrap
def fused_conv_unforward(x, weight):
    # Input: x [1, 256, 32, 32], weight [128, 256, 1, 1]
    # Output: [1, 512, 256]
    
    n_channels_in = x.shape[1]
    n_channels_out = weight.shape[0]
    feat_h = x.shape[2]
    feat_w = x.shape[3]
    
    # Calculate unfolded dimensions: 32x32 with 2x2 kernel and stride 2 -> 16x16
    unfold_h = (feat_h - 2) // 2 + 1  # 16
    unfold_w = (feat_w - 2) // 2 + 1  # 16
    
    # Output shape: [1, 4*128, 16*16] = [1, 512, 256]
    out_shape = (1, 4 * n_channels_out, unfold_h * unfold_w)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Configure block size for optimal GPU occupancy
    BLOCK_SIZE = 1024  # Elements per kernel launch
    
    # Total output elements to process
    total_elements = n_channels_out * unfold_h * unfold_w
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_conv_unfold_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        out_ptr=out,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        feat_h=feat_h,
        feat_w=feat_w,
        unfold_h=unfold_h,
        unfold_w=unfold_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_conv_unforward