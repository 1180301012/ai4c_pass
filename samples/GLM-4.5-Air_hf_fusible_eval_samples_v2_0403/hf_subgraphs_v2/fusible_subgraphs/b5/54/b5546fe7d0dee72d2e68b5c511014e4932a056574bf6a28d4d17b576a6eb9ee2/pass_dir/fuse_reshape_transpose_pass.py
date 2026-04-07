import torch
import triton
import triton.language as tl

def pattern(in_2, in_3, conv2d):
    tmp_3 = torch.cat([in_2, in_3, conv2d], dim=1)
    tmp_4 = tmp_3.reshape(1, 8, 19, 196)
    tmp_5 = tmp_4.transpose(-1, -2)
    return tmp_5

def replacement_args(in_2, in_3, conv2d):
    return (in_2, in_3, conv2d)

@triton.jit
def fused_reshape_transpose_kernel(
    in2_ptr, in3_ptr, conv_ptr,
    out_ptr,
    batch_size, C_in2, H_in2, W_in2,
    C_in3, H_in3, W_in3,
    C_conv, H_conv, W_conv,
    C_out, H_out, W_out,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    grid_size = tl.cdiv(batch_size * C_out * H_out * W_out, BLOCK_SIZE)
    if pid >= grid_size:
        return
    
    # Calculate offsets for this thread
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < (batch_size * C_out * H_out * W_out)
    
    if not mask.any():
        return
        
    # Convert offset to multi-dimensional indices
    total_elements = offset
    w_idx = total_elements % W_out
    h_idx = (total_elements // W_out) % H_out
    c_idx = (total_elements // (W_out * H_out)) % C_out
    b_idx = total_elements // (W_out * H_out * C_out)
    
    # Initialize output accumulation
    out_vals = tl.zeros((BLOCK_SIZE,), dtype=out_ptr.type.element_ty)
    
    # Process each input tensor and fuse the operations
    for i_idx in range(C_out):
        # Determine which input this channel comes from
        if i_idx < C_in2:
            # Input 2 contribution
            in2_offset = b_idx * C_in2 * H_in2 * W_in2 + i_idx * H_in2 * W_in2
            h_idx_orig = h_idx
            w_idx_orig = w_idx
        elif i_idx < C_in2 + C_in3:
            # Input 3 contribution  
            in3_offset = b_idx * C_in3 * H_in3 * W_in3 + (i_idx - C_in2) * H_in3 * W_in3
            h_idx_orig = h_idx
            w_idx_orig = w_idx
        else:
            # Conv output contribution
            conv_offset = b_idx * C_conv * H_conv * W_conv + (i_idx - C_in2 - C_in3) * H_conv * W_conv
            h_idx_orig = h_idx
            w_idx_orig = w_idx
        
        # Reshape from original format and transpose
        if i_idx < C_in2:
            val = tl.load(in2_ptr + (in2_offset + h_idx_orig * W_in2 + w_idx_orig), mask=(w_idx_orig < W_in2) & (h_idx_orig < H_in2), other=0.0)
        elif i_idx < C_in2 + C_in3:
            val = tl.load(in3_ptr + (in3_offset + h_idx_orig * W_in3 + w_idx_orig), mask=(w_idx_orig < W_in3) & (h_idx_orig < H_in3), other=0.0)
        else:
            val = tl.load(conv_ptr + (conv_offset + h_idx_orig * W_conv + w_idx_orig), mask=(w_idx_orig < W_conv) & (h_idx_orig < H_conv), other=0.0)
        
        # Store transposed result (original w becomes h, h becomes w)
        out_idx = b_idx * C_out * H_out * W_out + i_idx * H_out * W_out + w_idx * H_out + h_idx
        idx_in_block = offset - (pid * BLOCK_SIZE)
        mask_val = idx_in_block < BLOCK_SIZE
        tl.store(out_ptr + out_idx, val, mask=(idx_in_block < BLOCK_SIZE) & (w_idx < W_out) & (h_idx < H_out))

@torch.fx.wrap  
def fused_reshape_transpose(in_2, in_3, conv2d):
    # Get input dimensions
    N2, C_in2, H_in2, W_in2 = in_2.shape
    N3, C_in3, H_in3, W_in3 = in_3.shape  
    N_conv, C_conv, H_conv, W_conv = conv2d.shape
    
    # The output of cat is batch=1, channels=C_in2+C_in3+C_conv, height=max(H_in2,H_in3,H_conv), width=max(W_in2,W_in3,W_conv)
    # Then reshape to (1, 8, 19, 196) and transpose last two dimensions
    # So we need to map to 8 channels, 19x196 spatial dimensions
    H_out = 196  # After transpose, this becomes the width dimension
    W_out = 19   # After transpose, this becomes the height dimension  
    C_out = 8
    
    # Create output tensor with transposed dimensions
    out = torch.empty((1, C_out, H_out, W_out), dtype=conv2d.dtype, device=conv2d.device)
    
    # Launch kernel
    total_elements = 1 * C_out * H_out * W_out
    BLOCK_SIZE = 1024
    
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_reshape_transpose_kernel[grid_size](
        in2_ptr=in_2,
        in3_ptr=in_3,
        conv_ptr=conv2d,
        out_ptr=out,
        batch_size=1,
        C_in2=C_in2, H_in2=H_in2, W_in2=W_in2,
        C_in3=C_in3, H_in3=H_in3, W_in3=W_in3,
        C_conv=C_conv, H_conv=H_conv, W_conv=W_conv,
        C_out=C_out, H_out=H_out, W_out=W_out,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_reshape_transpose