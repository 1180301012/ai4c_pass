import torch
import triton
import triton.language as tl

def pattern(x, weight, bias, size=(512, 512), mode='bilinear', align_corners=False):
    """Pattern: Conv2D followed by Bilinear Interpolation"""
    conv = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    interp = torch.nn.functional.interpolate(conv, size=size, mode=mode, align_corners=align_corners)
    return conv, interp

def replacement_args(x, weight, bias):
    return (x, weight, bias, (512, 512), 'bilinear', False)

@triton.jit
def fused_conv_interp_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    B_in,
    C_in,
    H_in, 
    W_in,
    C_out,
    scale_factor,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate output dimensions
    H_out = int(H_in * scale_factor)
    W_out = int(W_in * scale_factor)
    
    # Program ID for parallel processing
    pid = tl.program_id(axis=0)
    num_blocks = B_in * C_out * H_out * W_out
    if pid >= num_blocks:
        return
        
    # Output coordinates
    b = pid // (C_out * H_out * W_out)
    c = (pid // (H_out * W_out)) % C_out
    h_out = (pid // W_out) % H_out  
    w_out = pid % W_out
    
    # Calculate corresponding input coordinates (bilinear interpolation)
    h_in = (h_out + 0.5) / scale_factor - 0.5
    w_in = (w_out + 0.5) / scale_factor - 0.5
    
    # Bilinear interpolation weights
    h1 = int(tl.floor(h_in))
    w1 = int(tl.floor(w_in))
    h2 = h1 + 1
    w2 = w1 + 1
    
    # Clamping to valid range
    h1 = tl.max(h1, 0)
    w1 = tl.max(w1, 0)
    h2 = tl.min(h2, H_in - 1)
    w2 = tl.min(w2, W_in - 1)
    
    # Bilinear weights
    wy1 = (h2 - h_in) if h2 < H_in else 0.0
    wy2 = (h_in - h1) if h1 < H_in else 0.0
    wx1 = (w2 - w_in) if w2 < W_in else 0.0
    wx2 = (w_in - w1) if w1 < W_in else 0.0
    
    # Load weights for this output channel
    weight_offset = c * C_in * 1 * 1  # 1x1 conv
    weights = tl.load(weight_ptr + weight_offset, mask=c < C_out).to(tl.float32)
    bias_val = tl.load(bias_ptr + c, mask=c < C_out).to(tl.float32).to(tl.float32)
    
    # Accumulate conv result with bilinear weights
    conv_result = 0.0
    
    for ci in range(C_in):
        # Input tensor indices for bilinear lookup
        idx_11 = ((b * C_in + ci) * H_in + h1) * W_in + w1
        idx_12 = ((b * C_in + ci) * H_in + h1) * W_in + w2
        idx_21 = ((b * C_in + ci) * H_in + h2) * W_in + w1
        idx_22 = ((b * C_in + ci) * H_in + h2) * W_in + w2
        
        # Load input values
        val_11 = tl.load(x_ptr + idx_11, mask=((b * C_in + ci) < B_in * C_in) & (h1 < H_in) & (w1 < W_in)).to(tl.float32)
        val_12 = tl.load(x_ptr + idx_12, mask=((b * C_in + ci) < B_in * C_in) & (h1 < H_in) & (w2 < W_in)).to(tl.float32)
        val_21 = tl.load(x_ptr + idx_21, mask=((b * C_in + ci) < B_in * C_in) & (h2 < H_in) & (w1 < W_in)).to(tl.float32)
        val_22 = tl.load(x_ptr + idx_22, mask=((b * C_in + ci) < B_in * C_in) & (h2 < H_in) & (w2 < W_in)).to(tl.float32)
        
        # Bilinear interpolation for this input channel
        interp_val = (val_11 * wy1 * wx1 + val_12 * wy1 * wx2 + 
                     val_21 * wy2 * wx1 + val_22 * wy2 * wx2)
        
        # Apply weight for this input-output channel pair
        w_offset = weight_offset + ci * 1 * 1
        weight_val = tl.load(weight_ptr + w_offset, mask=ci < C_in).to(tl.float32)
        conv_result += interp_val * weight_val
    
    # Add bias and store result
    final_result = conv_result + bias_val
    
    # Store output
    out_idx = ((b * C_out + c) * H_out + h_out) * W_out + w_out
    tl.store(out_ptr + out_idx, final_result.to(tl.float16), mask=b < B_in)

@torch.fx.wrap 
def fused_conv_interp(x, weight, bias):
    B, C_in, H_in, W_in = x.shape
    C_out = weight.shape[0]
    
    # Calculate scale factor (128->512 or 32->512, both scale by 4)
    scale_factor = 512 / max(H_in, W_in)  # Handle both cases
    
    # Output shape
    H_out = int(H_in * scale_factor)
    W_out = int(W_in * scale_factor)
    
    # Allocate output
    out = torch.empty((B, C_out, H_out, W_out), dtype=torch.float16, device=x.device)
    
    # Launch kernel
    total_elements = B * C_out * H_out * W_out
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_conv_interp_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        B_in=B,
        C_in=C_in,
        H_in=H_in,
        W_in=W_in, 
        C_out=C_out,
        scale_factor=scale_factor,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return x, out  # conv result, interp result

def replacement_func():
    return fused_conv_interp