import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
    """Pattern matching the full computation graph"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = in_4
    tmp_5 = in_5
    tmp_6 = in_6
    tmp_7 = torch.conv2d(in_8, tmp_2, tmp_1, (1, 1), (0, 0), (1, 1), 1)
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
    tmp_9 = tmp_8 * tmp_0
    tmp_10 = in_7 + tmp_9
    tmp_11 = torch.nn.functional.batch_norm(tmp_10, tmp_3, tmp_4, tmp_6, tmp_5, False, 0.1, 1e-05)
    return (tmp_11, tmp_10)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_C_OUT': 64, 'BLOCK_SIZE_SPATIAL': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_C_OUT': 64, 'BLOCK_SIZE_SPATIAL': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE_C_OUT': 32, 'BLOCK_SIZE_SPATIAL': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_C_OUT': 32, 'BLOCK_SIZE_SPATIAL': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE_C_OUT': 64, 'BLOCK_SIZE_SPATIAL': 128}, num_warps=4),
    ],
    key=['C_OUT', 'C_IN', 'SPATIAL'],
)
@triton.jit
def fused_conv1x1_layerscale_residual_bn_kernel(
    x_ptr, weight_ptr, bias_ptr, layer_scale_ptr, residual_ptr,
    bn_weight_ptr, bn_bias_ptr, bn_mean_ptr, bn_var_ptr,
    out_bn_ptr, out_residual_ptr,
    B, C_IN, C_OUT, SPATIAL,
    eps: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):
    """
    Fused kernel for 1x1 conv + layer scale + residual + batch norm
    """
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_cout = tl.program_id(1)
    pid_spatial = tl.program_id(2)
    
    # Compute offsets
    c_out_start = pid_cout * BLOCK_SIZE_C_OUT
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_SIZE_C_OUT)
    c_out_mask = c_out_offsets < C_OUT
    
    spatial_start = pid_spatial * BLOCK_SIZE_SPATIAL
    spatial_offsets = spatial_start + tl.arange(0, BLOCK_SIZE_SPATIAL)
    spatial_mask = spatial_offsets < SPATIAL
    
    # Load batch norm parameters
    bn_weight = tl.load(bn_weight_ptr + c_out_offsets, mask=c_out_mask, other=0.0)
    bn_bias = tl.load(bn_bias_ptr + c_out_offsets, mask=c_out_mask, other=0.0)
    bn_mean = tl.load(bn_mean_ptr + c_out_offsets, mask=c_out_mask, other=0.0)
    bn_var = tl.load(bn_var_ptr + c_out_offsets, mask=c_out_mask, other=0.0)
    bn_std = tl.sqrt(bn_var + eps)
    
    # Load layer scale parameter
    layer_scale = tl.load(layer_scale_ptr + c_out_offsets, mask=c_out_mask, other=0.0)
    
    # Load conv bias
    conv_bias = tl.load(bias_ptr + c_out_offsets, mask=c_out_mask, other=0.0)
    
    # Process each spatial location
    for s_idx in range(BLOCK_SIZE_SPATIAL):
        s = spatial_start + s_idx
        if s >= SPATIAL:
            break
        
        # Initialize accumulator for conv output
        acc = tl.zeros([BLOCK_SIZE_C_OUT], dtype=tl.float32)
        
        # Perform 1x1 convolution (matrix multiply over channels)
        for c_in in range(C_IN):
            # Load input
            x_idx = pid_b * C_IN * SPATIAL + c_in * SPATIAL + s
            x_val = tl.load(x_ptr + x_idx)
            
            # Load weight [C_OUT, C_IN, 1, 1]
            weight_idx = c_out_offsets * C_IN + c_in
            weight_val = tl.load(weight_ptr + weight_idx, mask=c_out_mask, other=0.0)
            
            acc += x_val * weight_val
        
        # Add conv bias
        conv_out = acc + conv_bias
        
        # Apply layer scale (dropout with p=0 is identity, so skip it)
        scaled = conv_out * layer_scale
        
        # Add residual
        residual_idx = pid_b * C_OUT * SPATIAL + c_out_offsets * SPATIAL + s
        residual_val = tl.load(residual_ptr + residual_idx, mask=c_out_mask, other=0.0)
        residual_out = residual_val + scaled
        
        # Apply batch norm
        bn_out = (residual_out - bn_mean) / bn_std * bn_weight + bn_bias
        
        # Store results
        out_bn_idx = pid_b * C_OUT * SPATIAL + c_out_offsets * SPATIAL + s
        tl.store(out_bn_ptr + out_bn_idx, bn_out, mask=c_out_mask)
        tl.store(out_residual_ptr + out_bn_idx, residual_out, mask=c_out_mask)


@torch.fx.wrap
def fused_conv1x1_layerscale_residual_bn(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
    """
    Args:
        in_0: layer_scale [C_OUT, 1, 1]
        in_1: conv_bias [C_OUT]
        in_2: conv_weight [C_OUT, C_IN, 1, 1]
        in_3: bn_running_mean [C_OUT]
        in_4: bn_running_var [C_OUT]
        in_5: bn_bias [C_OUT]
        in_6: bn_weight [C_OUT]
        in_7: residual [B, C_OUT, H, W]
        in_8: input [B, C_IN, H, W]
    """
    B, C_IN, H, W = in_8.shape
    C_OUT = in_2.shape[0]
    SPATIAL = H * W
    
    # Allocate output tensors
    out_bn = torch.empty((B, C_OUT, H, W), device=in_8.device, dtype=in_8.dtype)
    out_residual = torch.empty((B, C_OUT, H, W), device=in_8.device, dtype=in_8.dtype)
    
    # Reshape layer_scale to 1D
    layer_scale = in_0.view(-1)
    
    # Reshape inputs for easier indexing
    x_flat = in_8.contiguous()
    residual_flat = in_7.contiguous()
    weight_flat = in_2.contiguous()
    
    # Grid configuration
    BLOCK_SIZE_C_OUT = 64
    BLOCK_SIZE_SPATIAL = 256
    grid = (B, triton.cdiv(C_OUT, BLOCK_SIZE_C_OUT), triton.cdiv(SPATIAL, BLOCK_SIZE_SPATIAL))
    
    fused_conv1x1_layerscale_residual_bn_kernel[grid](
        x_flat, weight_flat, in_1, layer_scale, residual_flat,
        in_6, in_5, in_3, in_4,
        out_bn, out_residual,
        B, C_IN, C_OUT, SPATIAL,
        eps=1e-05,
    )
    
    return (out_bn, out_residual)


def replacement_func():
    return fused_conv1x1_layerscale_residual_bn