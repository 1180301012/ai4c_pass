import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'TILE_M': 16, 'TILE_N': 16, 'TILE_K': 32}, num_warps=2),
        triton.Config({'TILE_M': 16, 'TILE_N': 32, 'TILE_K': 32}, num_warps=2),
        triton.Config({'TILE_M': 32, 'TILE_N': 16, 'TILE_K': 32}, num_warps=4),
        triton.Config({'TILE_M': 32, 'TILE_N': 32, 'TILE_K': 32}, num_warps=4),
        triton.Config({'TILE_M': 64, 'TILE_N': 32, 'TILE_K': 32}, num_warps=4),
        triton.Config({'TILE_M': 64, 'TILE_N': 64, 'TILE_K': 32}, num_warps=4),
        triton.Config({'TILE_M': 128, 'TILE_N': 16, 'TILE_K': 32}, num_warps=8),
        triton.Config({'TILE_M': 128, 'TILE_N': 32, 'TILE_K': 32}, num_warps=8),
        triton.Config({'TILE_M': 128, 'TILE_N': 64, 'TILE_K': 32}, num_warps=8),
        triton.Config({'TILE_M': 16, 'TILE_N': 16, 'TILE_K': 64}, num_warps=2),
        triton.Config({'TILE_M': 16, 'TILE_N': 32, 'TILE_K': 64}, num_warps=4),
        triton.Config({'TILE_M': 32, 'TILE_N': 32, 'TILE_K': 64}, num_warps=4),
        triton.Config({'TILE_M': 64, 'TILE_N': 32, 'TILE_K': 64}, num_warps=4),
        triton.Config({'TILE_M': 64, 'TILE_N': 64, 'TILE_K': 64}, num_warps=8),
        triton.Config({'TILE_M': 128, 'TILE_N': 32, 'TILE_K': 64}, num_warps=8),
    ],
    key=['C_in', 'C_out'],
)
@triton.jit
def fused_conv_bn_add_kernel(
    input_ptr, weight_ptr, bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    shortcut_ptr, output_ptr,
    M, C_in, C_out, H, W,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_wt_0, stride_wt_1,
    stride_sc_n, stride_sc_c, stride_sc_h, stride_sc_w,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """Fused 1x1 Conv2d + BatchNorm (eval) + Residual Add kernel.
    
    Computes: output = BN(Conv1x1(input, weight)) + shortcut
    
    For a 1x1 conv, this is equivalent to a matrix multiplication:
    output[spatial, c_out] = sum_{c_in} input[spatial, c_in] * weight[c_out, c_in]
    then apply BN per channel, then add shortcut.
    """
    # Program IDs - 1D grid mapping to 2D tiles
    pid = tl.program_id(0)
    num_n_tiles = tl.cdiv(C_out, TILE_N)
    pid_m = pid // num_n_tiles
    pid_n = pid % num_n_tiles
    
    # Tile offsets for spatial positions and output channels
    m_start = pid_m * TILE_M
    n_start = pid_n * TILE_N
    
    m_offsets = m_start + tl.arange(0, TILE_M)
    n_offsets = n_start + tl.arange(0, TILE_N)
    
    m_mask = m_offsets < M
    n_mask = n_offsets < C_out
    
    # Decompose flat spatial index into (batch, height, width)
    HW = H * W
    n_batch = m_offsets // HW
    hw = m_offsets % HW
    h_idx = hw // W
    w_idx = hw % W
    
    # Compute base offsets for each spatial position in different tensors
    spatial_base_in = n_batch * stride_in_n + h_idx * stride_in_h + w_idx * stride_in_w
    spatial_base_sc = n_batch * stride_sc_n + h_idx * stride_sc_h + w_idx * stride_sc_w
    spatial_base_out = n_batch * stride_out_n + h_idx * stride_out_h + w_idx * stride_out_w
    
    # Load BN parameters for this tile of output channels (compute in float32)
    bn_mean_vals = tl.load(bn_mean_ptr + n_offsets, mask=n_mask, other=0.0).to(tl.float32)
    bn_var_vals = tl.load(bn_var_ptr + n_offsets, mask=n_mask, other=1.0).to(tl.float32)
    bn_weight_vals = tl.load(bn_weight_ptr + n_offsets, mask=n_mask, other=1.0).to(tl.float32)
    bn_bias_vals = tl.load(bn_bias_ptr + n_offsets, mask=n_mask, other=0.0).to(tl.float32)
    
    # Compute BN scale and shift (fusing BN into affine transform)
    # BN formula: output = weight * (x - mean) / sqrt(var + eps) + bias
    # = (weight / sqrt(var + eps)) * x + (bias - weight * mean / sqrt(var + eps))
    # = scale * x + shift
    bn_scale = bn_weight_vals / tl.sqrt(bn_var_vals + 1e-5)  # [TILE_N]
    bn_shift = bn_bias_vals - bn_scale * bn_mean_vals  # [TILE_N]
    
    # Accumulator for conv output [TILE_M, TILE_N] - float32 for precision
    acc = tl.zeros([TILE_M, TILE_N], dtype=tl.float32)
    
    # Loop over input channels in tiles
    for k_start in range(0, C_in, TILE_K):
        k_offsets = k_start + tl.arange(0, TILE_K)
        k_mask = k_offsets < C_in
        
        # Load input tile: [TILE_M, TILE_K]
        # input[n, c_in, h, w] at spatial_base + c_in * stride_in_c
        in_offsets = spatial_base_in[:, None] + k_offsets[None, :] * stride_in_c
        input_mask = m_mask[:, None] & k_mask[None, :]
        input_tile = tl.load(input_ptr + in_offsets, mask=input_mask, other=0.0).to(tl.float32)
        
        # Load weight tile: [TILE_N, TILE_K]
        # weight[c_out, c_in] at c_out * stride_wt_0 + c_in * stride_wt_1
        wt_offsets = n_offsets[:, None] * stride_wt_0 + k_offsets[None, :] * stride_wt_1
        weight_mask = n_mask[:, None] & k_mask[None, :]
        weight_tile = tl.load(weight_ptr + wt_offsets, mask=weight_mask, other=0.0).to(tl.float32)
        
        # Matrix multiply: [TILE_M, TILE_K] @ [TILE_K, TILE_N] -> [TILE_M, TILE_N]
        # weight_tile is [TILE_N, TILE_K], transpose to [TILE_K, TILE_N]
        weight_tile_T = tl.trans(weight_tile)
        acc += tl.dot(input_tile, weight_tile_T, allow_tf32=False)
    
    # Apply fused BN: scale * conv_output + shift
    output_vals = acc * bn_scale[None, :] + bn_shift[None, :]
    
    # Load shortcut tile: [TILE_M, TILE_N]
    # shortcut[n, c_out, h, w] at spatial_base + c_out * stride_sc_c
    sc_offsets = spatial_base_sc[:, None] + n_offsets[None, :] * stride_sc_c
    sc_mask = m_mask[:, None] & n_mask[None, :]
    shortcut_vals = tl.load(shortcut_ptr + sc_offsets, mask=sc_mask, other=0.0).to(tl.float32)
    
    # Add shortcut
    output_vals = output_vals + shortcut_vals
    
    # Cast to output dtype and store
    output_vals_casted = output_vals.to(OUT_DTYPE)
    out_offsets = spatial_base_out[:, None] + n_offsets[None, :] * stride_out_c
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_offsets, output_vals_casted, mask=out_mask)


def _fused_conv_bn_add_impl(conv_input, conv_weight, bn_mean, bn_var, bn_weight, bn_bias, shortcut):
    """Implementation of fused conv + BN + add using Triton kernel."""
    N_batch, C_in, H, W = conv_input.shape
    C_out = conv_weight.shape[0]
    M = N_batch * H * W
    
    # Allocate output with same shape and dtype as shortcut
    output = torch.empty_like(shortcut)
    
    # Determine output dtype for Triton constexpr
    if output.dtype == torch.float16:
        out_dtype = tl.float16
    elif output.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    else:
        out_dtype = tl.float32
    
    # Get strides
    s_in = conv_input.stride()
    s_wt = conv_weight.stride()
    s_sc = shortcut.stride()
    s_out = output.stride()
    
    # Grid function for autotuning
    grid = lambda META: (
        triton.cdiv(M, META['TILE_M']) * triton.cdiv(C_out, META['TILE_N']),
    )
    
    fused_conv_bn_add_kernel[grid](
        input_ptr=conv_input, weight_ptr=conv_weight,
        bn_mean_ptr=bn_mean, bn_var_ptr=bn_var, 
        bn_weight_ptr=bn_weight, bn_bias_ptr=bn_bias,
        shortcut_ptr=shortcut, output_ptr=output,
        M=M, C_in=C_in, C_out=C_out, H=H, W=W,
        stride_in_n=s_in[0], stride_in_c=s_in[1], stride_in_h=s_in[2], stride_in_w=s_in[3],
        stride_wt_0=s_wt[0], stride_wt_1=s_wt[1],
        stride_sc_n=s_sc[0], stride_sc_c=s_sc[1], stride_sc_h=s_sc[2], stride_sc_w=s_sc[3],
        stride_out_n=s_out[0], stride_out_c=s_out[1], stride_out_h=s_out[2], stride_out_w=s_out[3],
        OUT_DTYPE=out_dtype,
    )
    
    return output