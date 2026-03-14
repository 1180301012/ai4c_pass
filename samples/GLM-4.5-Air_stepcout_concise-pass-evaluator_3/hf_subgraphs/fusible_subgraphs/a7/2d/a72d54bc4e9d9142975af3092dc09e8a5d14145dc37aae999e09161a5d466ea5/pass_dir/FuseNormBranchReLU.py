import torch
import triton
import triton.language as tl

def pattern(in_5, in_1, in_2, in_4, in_3):
    """Pattern matching for adaptive_avg_pool2d + batch_norm + ReLU fusion"""
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(in_5, (1, 1))
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=True)
    return tmp_8

def replacement_args(in_5, in_1, in_2, in_4, in_3):
    return (in_5, in_1, in_2, in_4, in_3)

@triton.jit
def norm_relu_fused_kernel(
    x_ptr,          # Input tensor [B, C, H, W]
    running_mean_ptr,  # Running mean [C]
    running_var_ptr,   # Running var [C] 
    weight_ptr,        # BN weight [C]
    bias_ptr,          # BN bias [C]
    out_ptr,           # Output [B, C, 1, 1]
    N,                 # Batch size
    C,                 # Channels
    H,                 # Height
    W,                 # Width
    eps: tl.constexpr, # Epsilon for BN
    BLOCK_SIZE_C: tl.constexpr
):
    # Each program handles a channel
    c = tl.program_id(0)
    
    # Load BN parameters for this channel
    running_mean = tl.load(running_mean_ptr + c)
    running_var = tl.load(running_var_ptr + c)
    weight = tl.load(weight_ptr + c)
    bias = tl.load(bias_ptr + c)
    
    # Compute spatial mean over H and W using vectorized loads
    spatial_sum = 0.0
    spatial_count = H * W  # Total number of spatial positions
    
    # Vectorized load for better memory efficiency
    for h in range(H):
        for w in range(W):
            # Compute offset for current spatial position: [B, C, H, W] layout
            offset = (n * C * H * W) + (c * H * W) + (h * W) + w
            val = tl.load(x_ptr + offset)
            spatial_sum += val
    
    # Compute mean
    spatial_mean = spatial_sum / spatial_count
    
    # Normalize and apply BN
    var_inv = 1.0 / tl.sqrt(running_var + eps)
    normalized = (spatial_mean - running_mean) * var_inv
    bn_out = normalized * weight + bias
    
    # Apply ReLU
    relu_out = tl.maximum(bn_out, 0.0)
    
    # Store result (output is [B, C, 1, 1])
    output_offset = (n * C + c) * 1 * 1
    tl.store(out_ptr + output_offset, relu_out)

@torch.fx.wrap
def fused_norm_relu(in_5, in_1, in_2, in_4, in_3):
    B, C, H, W = in_5.shape
    eps = 1e-05
    
    output_shape = (B, C, 1, 1)
    out = torch.empty(output_shape, dtype=in_5.dtype, device=in_5.device)
    
    # Grid: one program per channel
    grid = lambda meta: (C,)
    
    for n in range(B):
        norm_relu_fused_kernel[grid](
            x_ptr=in_5,
            running_mean_ptr=in_1,
            running_var_ptr=in_2,
            weight_ptr=in_4,
            bias_ptr=in_3,
            out_ptr=out,
            n=n,
            N=B,
            C=C,
            H=H,
            W=W,
            eps=eps,
            BLOCK_SIZE_C=256
        )
    
    return out

def replacement_func():
    return fused_norm_relu