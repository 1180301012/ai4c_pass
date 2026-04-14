import torch
import triton
import triton.language as tl

def pattern(in_6, in_4, in_0, in_1, in_3, in_2, in_5):
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.leaky_relu(tmp_6, 0.01, True)
    tmp_8 = tmp_7 + in_5
    return tmp_8

def replacement_args(in_6, in_4, in_0, in_1, in_3, in_2, in_5):
    return (in_6, in_4, in_0, in_1, in_3, in_2, in_5, "conv_bn_relu_residual_fusion")

@triton.jit
def conv_bn_relu_residual_kernel(
    x_ptr,                    # Input tensor pointer [B, C_in, H, W]
    weight_ptr,              # Conv weights [C_out, C_in, K, K]
    running_mean_ptr,        # BatchNorm running mean [C_out]
    running_var_ptr,         # BatchNorm running var [C_out]
    weight_bn_ptr,           # BatchNorm weight (gamma) [C_out]
    bias_bn_ptr,             # BatchNorm bias (beta) [C_out]
    residual_ptr,            # Residual tensor [B, C_out, H_out, W_out]
    out_ptr,                 # Output tensor [B, C_out, H_out, W_out]
    B: tl.constexpr,         # Batch size
    C_in: tl.constexpr,      # Input channels
    C_out: tl.constexpr,     # Output channels
    H: tl.constexpr,         # Input height
    W: tl.constexpr,         # Input width
    H_out: tl.constexpr,     # Output height
    W_out: tl.constexpr,     # Output width
    K: tl.constexpr,         # Kernel size
    BLOCK_SIZE: tl.constexpr, # Block size for better parallelism
):
    # Use 2D program grid: batch, channel, and spatial blocks
    batch_idx = tl.program_id(1)
    channel_idx = tl.program_id(0)
    
    # Calculate spatial block offsets
    block_pid = tl.program_id(2)
    spatial_blocks_per_dim = (H_out * W_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    h_block_idx = block_pid // spatial_blocks_per_dim
    w_block_idx = block_pid % spatial_blocks_per_dim
    
    # Calculate spatial range for this block
    h_start = h_block_idx * BLOCK_SIZE
    w_start = w_block_idx * BLOCK_SIZE
    h_end = min(h_start + BLOCK_SIZE, H_out)
    w_end = min(w_start + BLOCK_SIZE, W_out)
    
    # Skip if this block is out of bounds
    if batch_idx >= B or h_start >= H_out or w_start >= W_out:
        return
    
    # Compute each spatial position in this block
    for h_out in range(h_start, h_end):
        for w_out in range(w_start, w_end):
            # Convolution computation for this spatial position
            conv_val = 0.0
            for ci in range(C_in):
                for kh in range(K):
                    for kw in range(K):
                        # Calculate input coordinates with stride=1, padding=1
                        in_h = h_out - 1 + kh  # padding=1 means we start h_out-1
                        in_w = w_out - 1 + kw  # padding=1 means we start w_out-1
                        
                        # Check if input coordinates are valid
                        if (0 <= in_h) & (in_h < H) & (0 <= in_w) & (in_w < W):
                            # Load input value
                            input_offset = batch_idx * C_in * H * W + ci * H * W + in_h * W + in_w
                            x_val = tl.load(x_ptr + input_offset)
                            
                            # Load weight value (always valid since bounds are checked in Python)
                            weight_offset = channel_idx * C_in * K * K + ci * K * K + kh * K + kw
                            weight_val = tl.load(weight_ptr + weight_offset)
                            
                            conv_val += x_val * weight_val
            
            # BatchNorm computation - these loads are always valid since channel_idx < C_out
            mean_val = tl.load(running_mean_ptr + channel_idx)
            var_val = tl.load(running_var_ptr + channel_idx)
            gamma_val = tl.load(weight_bn_ptr + channel_idx)
            beta_val = tl.load(bias_bn_ptr + channel_idx)
            
            # Cast to fp32 for higher precision math operations (required for sqrt)
            conv_val_fp32 = tl.cast(conv_val, tl.float32)
            mean_val_fp32 = tl.cast(mean_val, tl.float32)
            var_val_fp32 = tl.cast(var_val, tl.float32)
            gamma_val_fp32 = tl.cast(gamma_val, tl.float32)
            beta_val_fp32 = tl.cast(beta_val, tl.float32)
            
            # Apply BN: gamma * (x - mean) / sqrt(var + eps) + beta
            # Cast back to original dtype after computation
            bn_val_fp32 = gamma_val_fp32 * (conv_val_fp32 - mean_val_fp32) / tl.sqrt(var_val_fp32 + 1e-05) + beta_val_fp32
            bn_val = tl.cast(bn_val_fp32, conv_val.dtype)
            
            # Apply LeakyReLU: max(0, x) + negative_slope * min(0, x)
            relu_val = tl.where(bn_val > 0, bn_val, bn_val * 0.01)
            
            # Load residual value
            residual_offset = batch_idx * C_out * H_out * W_out + channel_idx * H_out * W_out + h_out * W_out + w_out
            residual_val = tl.load(residual_ptr + residual_offset)
            
            # Add residual
            output_val = relu_val + residual_val
            
            # Store result
            output_offset = batch_idx * C_out * H_out * W_out + channel_idx * H_out * W_out + h_out * W_out + w_out
            tl.store(out_ptr + output_offset, output_val)

@torch.fx.wrap
def shared_dispatch_wrapper(*args):
    """
    Shared dispatch wrapper that routes to different kernels based on the route string.
    The last argument is the route string, all others are the actual input tensors.
    """
    # Separate the route string from the input tensors
    route = args[-1]
    inputs = args[:-1]
    
    if route == "conv_bn_relu_residual_fusion":
        x, weight, running_mean, running_var, weight_bn, bias_bn, residual = inputs
        
        # Get tensor shapes
        B, C_in, H, W = x.shape
        C_out, _, K, _ = weight.shape
        B_res, C_res, H_res, W_res = residual.shape
        
        # For stride=(1,1) and padding=(1,1), output size equals input size
        H_out = H
        W_out = W
        
        # Verify dimensions match (sanity check)
        assert C_out == C_res, f"Output channels {C_out} must match residual channels {C_res}"
        assert H_out == H_res, f"Output height {H_out} must match residual height {H_res}"
        assert W_out == W_res, f"Output width {W_out} must match residual width {W_res}"
        
        # Create output tensor
        out = torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
        
        # Use efficient 3D grid layout
        BLOCK_SIZE = 16  # Optimal block size for spatial dimensions
        num_spatial_blocks = (H_out * W_out + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (C_out, B, num_spatial_blocks)
        
        # Launch kernel with 3D grid
        conv_bn_relu_residual_kernel[grid](
            x_ptr=x,
            weight_ptr=weight,
            running_mean_ptr=running_mean,
            running_var_ptr=running_var,
            weight_bn_ptr=weight_bn,
            bias_bn_ptr=bias_bn,
            residual_ptr=residual,
            out_ptr=out,
            B=B, C_in=C_in, C_out=C_out, H=H, W=W,
            H_out=H_out, W_out=W_out, K=K, BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out
    else:
        # Should not happen for this single-pass setup
        raise ValueError(f"Unknown route: {route}")

def replacement_func():
    return shared_dispatch_wrapper