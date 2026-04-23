import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv_bn_relu_kernel(
    # Input and output pointers
    x_ptr, w_ptr, bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    residual_ptr, out_ptr,
    # Strides
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc,
    # BN parameter strides (for per-channel params)
    stride_bn_mean, stride_bn_var, stride_bn_weight, stride_bn_bias,
    # Tensor dimensions
    N, C_in, H, W, C_out, slope: tl.constexpr
):
    """
    Fused Conv2D + BatchNorm + LeakyReLU kernel.
    
    Grid: (N, C_out, H)
    Each program processes a column of W elements for its output position.
    Uses loop tiling to process multiple W positions per program for better efficiency.
    """
    # Program position
    pid = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Output position
    n = pid
    c_out = pid_h
    ho = pid_w
    
    # Process multiple Wo positions in a tile for efficiency
    # Each program processes Wo, Wo+1, ..., Wo+BLOCK_W-1
    BLOCK_W = 8  # Tile size for W dimension
    
    # Base offset for this program
    n_base = n * stride_xn
    c_base = c_out * stride_wn
    ho_base = ho * stride_xh
    
    # Load BN parameters once per program (they're the same for all W positions)
    var_c = tl.load(bn_var_ptr + c_out * stride_bn_var)
    var_c_fp32 = var_c.to(tl.float32)
    inv_std = 1.0 / tl.sqrt(var_c_fp32 + 1e-05)
    inv_std = inv_std.to(var_c.type)
    
    bn_weight_c = tl.load(bn_weight_ptr + c_out * stride_bn_weight)
    bn_scale = bn_weight_c * inv_std
    
    bn_bias_c = tl.load(bn_bias_ptr + c_out * stride_bn_bias)
    bn_mean_c = tl.load(bn_mean_ptr + c_out * stride_bn_mean)
    fused_bias = (bn_bias_c - bn_mean_c) * inv_std * bn_weight_c
    
    # Process each Wo position in the tile
    for wo_offset in range(BLOCK_W):
        wo = pid_w * BLOCK_W + wo_offset  # Compute actual Wo position
        
        # Only process if wo is in bounds
        in_bounds = wo < W
        
        # Compute output offset
        out_offset = n * C_out * H * W + c_out * H * W + ho * W + wo
        res_offset = out_offset
        
        # Load residual (use mask for bounds checking)
        residual = tl.load(residual_ptr + res_offset, mask=in_bounds, other=0.0)
        
        # Convolution accumulation
        acc = 0.0
        
        # Iterate over input channels
        for ci in range(C_in):
            # Base offsets
            x_base_offset = n_base + ci * stride_xc
            w_base_offset = c_base + ci * stride_wc
            
            # kh=0 (h-1 row)
            # wo-1
            h_load = ho - 1
            w_load = wo - 1
            mask_h = (h_load >= 0) and (h_load < H)
            mask_w = (w_load >= 0) and (w_load < W)
            x_offset = x_base_offset + h_load * stride_xh + w_load * stride_xw
            x_val = tl.load(x_ptr + x_offset, mask=(mask_h and mask_w) and in_bounds, other=0.0)
            w_val = tl.load(w_ptr + w_base_offset + 0, mask=in_bounds, other=0.0)
            acc = acc + tl.where(in_bounds, x_val * w_val * bn_scale, 0.0)
            
            # wo
            x_offset = x_base_offset + h_load * stride_xh + wo * stride_xw
            x_val = tl.load(x_ptr + x_offset, mask=mask_h and in_bounds, other=0.0)
            w_val = tl.load(w_ptr + w_base_offset + 1, mask=in_bounds, other=0.0)
            acc = acc + tl.where(in_bounds, x_val * w_val * bn_scale, 0.0)
            
            # wo+1
            w_load = wo + 1
            mask_w = (w_load >= 0) and (w_load < W)
            x_offset = x_base_offset + h_load * stride_xh + w_load * stride_xw
            x_val = tl.load(x_ptr + x_offset, mask=(mask_h and mask_w) and in_bounds, other=0.0)
            w_val = tl.load(w_ptr + w_base_offset + 2, mask=in_bounds, other=0.0)
            acc = acc + tl.where(in_bounds, x_val * w_val * bn_scale, 0.0)
            
            # kh=1 (h row)
            # wo-1
            h_load = ho
            w_load = wo - 1
            mask_w = (w_load >= 0) and (w_load < W)
            x_offset = x_base_offset + h_load * stride_xh + w_load * stride_xw
            x_val = tl.load(x_ptr + x_offset, mask=mask_w and in_bounds, other=0.0)
            w_val = tl.load(w_ptr + w_base_offset + 3, mask=in_bounds, other=0.0)
            acc = acc + tl.where(in_bounds, x_val * w_val * bn_scale, 0.0)
            
            # wo
            x_offset = x_base_offset + ho * stride_xh + wo * stride_xw
            x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
            w_val = tl.load(w_ptr + w_base_offset + 4, mask=in_bounds, other=0.0)
            acc = acc + tl.where(in_bounds, x_val * w_val * bn_scale, 0.0)
            
            # wo+1
            w_load = wo + 1
            mask_w = (w_load >= 0) and (w_load < W)
            x_offset = x_base_offset + h_load * stride_xh + w_load * stride_xw
            x_val = tl.load(x_ptr + x_offset, mask=mask_w and in_bounds, other=0.0)
            w_val = tl.load(w_ptr + w_base_offset + 5, mask=in_bounds, other=0.0)
            acc = acc + tl.where(in_bounds, x_val * w_val * bn_scale, 0.0)
            
            # kh=2 (h+1 row)
            # wo-1
            h_load = ho + 1
            w_load = wo - 1
            mask_h = (h_load >= 0) and (h_load < H)
            mask_w = (w_load >= 0) and (w_load < W)
            x_offset = x_base_offset + h_load * stride_xh + w_load * stride_xw
            x_val = tl.load(x_ptr + x_offset, mask=(mask_h and mask_w) and in_bounds, other=0.0)
            w_val = tl.load(w_ptr + w_base_offset + 6, mask=in_bounds, other=0.0)
            acc = acc + tl.where(in_bounds, x_val * w_val * bn_scale, 0.0)
            
            # wo
            x_offset = x_base_offset + h_load * stride_xh + wo * stride_xw
            x_val = tl.load(x_ptr + x_offset, mask=mask_h and in_bounds, other=0.0)
            w_val = tl.load(w_ptr + w_base_offset + 7, mask=in_bounds, other=0.0)
            acc = acc + tl.where(in_bounds, x_val * w_val * bn_scale, 0.0)
            
            # wo+1
            w_load = wo + 1
            mask_w = (w_load >= 0) and (w_load < W)
            x_offset = x_base_offset + h_load * stride_xh + w_load * stride_xw
            x_val = tl.load(x_ptr + x_offset, mask=(mask_h and mask_w) and in_bounds, other=0.0)
            w_val = tl.load(w_ptr + w_base_offset + 8, mask=in_bounds, other=0.0)
            acc = acc + tl.where(in_bounds, x_val * w_val * bn_scale, 0.0)
        
        # Add bias and residual, apply activation
        result = acc + fused_bias + residual
        result = tl.where(result < 0, result * slope, result)
        
        # Store result with bounds check
        tl.store(out_ptr + out_offset, result, mask=in_bounds)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Pattern to match: Conv2D -> BatchNorm -> LeakyReLU -> Add
    
    Args:
        in_0: running_mean (shape [C])
        in_1: running_var (shape [C])
        in_2: bn_bias (shape [C])
        in_3: bn_weight (shape [C])
        in_4: conv_weight (shape [C_out, C_in, 3, 3])
        in_5: residual (shape [N, C_out, H, W])
        in_6: input (shape [N, C_in, H, W])
    """
    # Conv2D with padding=1, stride=1, dilation=1, groups=1
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    
    # BatchNorm with running stats (inference mode)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    
    # LeakyReLU activation
    tmp_7 = torch.nn.functional.leaky_relu(tmp_6, 0.01, True)
    
    # Add residual
    tmp_8 = tmp_7 + in_5
    
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """Extract arguments for the replacement function."""
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


@torch.fx.wrap
def fused_conv_bn_relu_wrapper(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Wrapper for the fused Conv2D + BatchNorm + LeakyReLU kernel.
    
    All BN computations are done inside the Triton kernel to avoid
    using tensor operations that are blocked by the framework.
    """
    # Get tensor info
    x = in_6
    N, C_in, H, W = x.shape
    C_out = in_4.shape[0]
    
    # Ensure input and weights are on GPU
    device = in_4.device
    x = x.to(device)
    
    # Ensure BN parameters are on GPU (they might be on CPU)
    bn_mean = in_0.to(device)
    bn_var = in_1.to(device)
    bn_weight = in_3.to(device)
    bn_bias = in_2.to(device)
    
    # Residual should also be on GPU
    residual = in_5.to(device)
    
    # Ensure weights are contiguous
    w = in_4.to(device).contiguous()
    
    # Compute grid dimensions
    # Use a coarser grid with fewer programs
    # Each program processes a tile of W elements
    BLOCK_W = 8
    grid_n = N
    grid_h = H
    grid_w = (W + BLOCK_W - 1) // BLOCK_W  # Number of tiles in W
    
    grid = (grid_n, grid_h, grid_w)
    
    # Allocate output
    out = torch.empty(N, C_out, H, W, dtype=x.dtype, device=device)
    
    # Get strides
    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    stride_wn = w.stride(0)
    stride_wc = w.stride(1)
    
    # BN parameter strides (assuming contiguous 1D tensors)
    stride_bn_mean = 1
    stride_bn_var = 1
    stride_bn_weight = 1
    stride_bn_bias = 1
    
    # Launch kernel
    fused_conv_bn_relu_kernel[grid](
        x, w, bn_mean, bn_var, bn_weight, bn_bias,
        residual, out,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wn, stride_wc,
        stride_bn_mean, stride_bn_var, stride_bn_weight, stride_bn_bias,
        N, C_in, H, W, C_out,
        slope=0.01
    )
    
    return out


def replacement_func():
    """Return the replacement function."""
    return fused_conv_bn_relu_wrapper