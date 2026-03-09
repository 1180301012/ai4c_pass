import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    # Interpolate operation
    result = torch.nn.functional.interpolate(input_tensor, size=(128, 128), mode='bilinear', align_corners=False)
    return result

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def bilinear_interpolate_kernel(
    input_ptr,    # Input tensor [B, C, H, W]
    output_ptr,   # Output tensor [B, C, H_out, W_out]
    B, C, H, W, H_out, W_out,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Each program handles a tile of the output
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h_out = tl.program_id(2)
    pid_w_out = tl.program_id(3)
    
    # Compute output coordinates
    h_out = pid_h_out * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_out = pid_w_out * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    h_out = h_out.to(tl.int32)
    w_out = w_out.to(tl.int32)
    
    # Create output masks
    h_out_mask = h_out < H_out
    w_out_mask = w_out < W_out
    out_mask = h_out_mask[:, None] & w_out_mask[None, :]
    
    # Scale input coordinates to input space
    scale_h = H / H_out
    scale_w = W / W_out
    
    # Compute source coordinates
    h_float = (h_out + 0.5) * scale_h - 0.5
    w_float = (w_out + 0.5) * scale_w - 0.5
    
    # Get integer coordinates and weights for bilinear interpolation
    h0 = tl.floor(h_float).to(tl.int32)
    w0 = tl.floor(w_float).to(tl.int32)
    h1 = h0 + 1
    w1 = w0 + 1
    
    # Clamp coordinates to valid range
    h0 = tl.maximum(0, tl.minimum(H - 1, h0))
    w0 = tl.maximum(0, tl.minimum(W - 1, w0))
    h1 = tl.maximum(0, tl.minimum(H - 1, h1))
    w1 = tl.maximum(0, tl.minimum(W - 1, w1))
    
    # Compute interpolation weights
    h_alpha = h_float - h0.to(tl.float32)
    w_alpha = w_float - w0.to(tl.float32)
    
    h_alpha_inv = 1.0 - h_alpha
    w_alpha_inv = 1.0 - w_alpha
    
    # Process each element in the block
    for c in range(0, BLOCK_SIZE_C):
        # Process each output coordinate
        for i, h in enumerate(h_out):
            for j, w in enumerate(w_out):
                if h_out_mask[i] and w_out_mask[j]:
                    # Bilinear interpolation
                    # Top-left sample
                    tl_val = tl.load(input_ptr + ((pid_b * C + c) * H + h0[i]) * W + w0[j])
                    top_left = tl_val * h_alpha_inv[i] * w_alpha_inv[j]
                    
                    # Top-right sample
                    tl_val = tl.load(input_ptr + ((pid_b * C + c) * H + h0[i]) * W + w1[j])
                    top_right = tl_val * h_alpha_inv[i] * w_alpha[j]
                    
                    # Bottom-left sample
                    tl_val = tl.load(input_ptr + ((pid_b * C + c) * H + h1[i]) * W + w0[j])
                    bottom_left = tl_val * h_alpha[i] * w_alpha_inv[j]
                    
                    # Bottom-right sample
                    tl_val = tl.load(input_ptr + ((pid_b * C + c) * H + h1[i]) * W + w1[j])
                    bottom_right = tl_val * h_alpha[i] * w_alpha[j]
                    
                    # Sum up the contributions
                    output_val = top_left + top_right + bottom_left + bottom_right
                    
                    # Store result
                    store_offset = ((pid_b * C + c) * H_out + h[i]) * W_out + w[j]
                    tl.store(output_ptr + store_offset, output_val)

@torch.fx.wrap
def optimized_bilinear_interpolate(input_tensor):
    B, C, H, W = input_tensor.shape
    H_out, W_out = 128, 128
    
    output = torch.empty((B, C, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Configure block sizes
    BLOCK_SIZE_C = 1
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    
    # Calculate grid size
    grid_c = triton.cdiv(C, BLOCK_SIZE_C)
    grid_h = triton.cdiv(H_out, BLOCK_SIZE_H)
    grid_w = triton.cdiv(W_out, BLOCK_SIZE_W)
    
    # Launch kernel
    bilinear_interpolate_kernel[(B, grid_c, grid_h, grid_w)](
        input_tensor,
        output,
        B, C, H, W, H_out, W_out,
        BLOCK_SIZE_C,
        BLOCK_SIZE_H,
        BLOCK_SIZE_W
    )
    
    return output

def replacement_func():
    return optimized_bilinear_interpolate