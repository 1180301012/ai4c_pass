import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    # Pattern 2: conv(in_7) -> + in_6 -> + in_7  
    # Use simpler conv2d call without groups parameter (defaults to 1)
    tmp_6 = torch.conv2d(in_7, in_5, in_4, (1, 1), (0, 0), (1, 1))
    tmp_7 = in_6 + tmp_6
    tmp_8 = tmp_7 + in_7
    tmp_9 = torch.nn.functional.batch_norm(tmp_8, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_10 = tmp_9.mean((2, 3), keepdim=True)
    return (tmp_9, tmp_10)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7)

@triton.jit
def fused_conv_add_add_bn_mean_kernel(
    # Input tensors
    in_6_ptr, in_7_ptr,  # Input activations [N, C, H, W]
    weight_ptr, bias_ptr,  # Conv weights [C, 1, 1, 1] and bias [C]
    running_mean_ptr, running_var_ptr,  # BN buffers [C]
    bn_weight_ptr, bn_bias_ptr,  # BN parameters [C]
    # Output tensors  
    out_ptr, out_mean_ptr,  # Output [N, C, H, W] and mean [N, C, 1, 1]
    # Metadata
    N, C, H, W,
    eps: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Compute program indices
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Compute ranges this program handles
    n_start = pid_n * BLOCK_SIZE_N
    c_start = pid_c * BLOCK_SIZE_C
    
    # Create masks for valid elements
    n_mask = n_start + tl.arange(0, BLOCK_SIZE_N) < N
    c_mask = c_start + tl.arange(0, BLOCK_SIZE_C) < C
    
    # Load conv weights and bias (broadcasted)
    weight = tl.load(weight_ptr + (c_start,) * 4, mask=c_mask, other=0.0)
    bias = tl.load(bias_ptr + c_start, mask=c_mask, other=0.0)
    weight = weight.to(tl.float32)
    bias = bias.to(tl.float32)
    
    # Load BN parameters
    bn_weight = tl.load(bn_weight_ptr + c_start, mask=c_mask, other=1.0)
    bn_bias = tl.load(bn_bias_ptr + c_start, mask=c_mask, other=0.0)
    running_mean = tl.load(running_mean_ptr + c_start, mask=c_mask, other=0.0)
    running_var = tl.load(running_var_ptr + c_start, mask=c_mask, other=1.0)
    
    # Pre-compute 1/(sqrt(variance + eps))
    inv_std = 1.0 / tl.sqrt(running_var + eps)
    bn_scale = bn_weight * inv_std
    bn_bias = bn_bias - running_mean * bn_scale
    
    # Process spatial dimensions
    for h in range(H):
        offsets_h = h * W
        
        for w in range(W):
            spatial_offset = offsets_h + w
            
            # Load input activations for all N channels in this block
            in_6_offsets = (n_start[:, None] * C * H * W + 
                           c_start[None, :] * H * W + 
                           spatial_offset[None, :] * C + 
                           tl.arange(0, BLOCK_SIZE_C)[None, :])
            
            in_6 = tl.load(in_6_ptr + in_6_offsets, mask=(n_mask[:, None] & c_mask[None, :]), other=0.0)
            in_7 = tl.load(in_7_ptr + in_6_offsets, mask=(n_mask[:, None] & c_mask[None, :]), other=0.0)
            
            # 1x1 convolution (element-wise multiplication with 1x1 weights + bias)
            conv_out = in_7 * weight[None, :] + bias[None, :]
            
            # Two additions: conv_out + in_6, then + in_7
            add_out = conv_out + in_6
            add_out = add_out + in_7
            
            # Batch normalization
            bn_out = add_out * bn_scale[None, :] + bn_bias[None, :]
            
            # Store output
            out_offsets = in_6_offsets
            tl.store(out_ptr + out_offsets, bn_out, mask=(n_mask[:, None] & c_mask[None, :]))
            
            # Accumulate mean values
            if h == 0 and w == 0:
                # For mean calculation, we'll accumulate in the first spatial position
                tl.store(out_mean_ptr + (n_start[:] * C + c_start[:]), 
                        bn_out.sum(0), mask=n_mask & c_mask)

@torch.fx.wrap
def fused_conv_add_add_bn_mean(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    # Get tensor shapes
    N, C, H, W = in_6.shape
    
    # Create output tensors
    out = torch.empty_like(in_6)
    out_mean = torch.empty((N, C, 1, 1), device=in_6.device, dtype=in_6.dtype)
    
    # Setup grid dimensions
    BLOCK_SIZE_N = 64  # Process multiple batches together
    BLOCK_SIZE_C = 256  # Process multiple channels together
    
    num_blocks_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_blocks_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Launch kernel
    fused_conv_add_add_bn_mean_kernel[(num_blocks_n, num_blocks_c)](
        in_6_ptr=in_6,
        in_7_ptr=in_7,
        weight_ptr=in_5,
        bias_ptr=in_4,
        running_mean_ptr=in_0,
        running_var_ptr=in_1,
        bn_weight_ptr=in_3,
        bn_bias_ptr=in_2,
        out_ptr=out,
        out_mean_ptr=out_mean.view(-1),  # Flatten for 1D pointer
        N=N, C=C, H=H, W=W,
        eps=1e-05,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return out, out_mean

def replacement_func():
    return fused_conv_add_add_bn_mean