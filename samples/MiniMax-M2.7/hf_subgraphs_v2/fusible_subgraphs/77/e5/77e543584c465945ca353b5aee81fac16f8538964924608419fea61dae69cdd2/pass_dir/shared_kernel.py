import torch
import triton
import triton.language as tl


# ============================================================================
# Shared Kernel for Fused Conv2D + Mean operations
# This file contains the Triton kernel implementation
# ============================================================================


@triton.jit
def fused_conv2d_mean_kernel_impl(
    # Input pointers
    input_ptr, weight_ptr,
    # Output pointers  
    output_ptr, mean_ptr,
    # Shapes
    B, C_out, H, W,
    stride_h, stride_w,
    # Compiled constants
    H_out: tl.constexpr, W_out: tl.constexpr, K: tl.constexpr, 
):
    """
    Fused Conv2D + Mean over spatial dimensions kernel.
    
    For depthwise-like convolution where weight shape is [C_out, 1, 3, 3].
    One program per (batch, channel).
    """
    # Grid: one program per batch * channel
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    
    n_spatial = H_out * W_out
    
    # Get batch and channel for this program
    b = pid // C_out
    c = pid % C_out
    
    if b >= B or c >= C_out:
        return
    
    # Compute convolution output for all spatial positions
    # and accumulate sum for mean
    
    # Initialize accumulator for mean
    mean_acc = tl.cast(0.0, tl.float32)
    
    # Load weight: shape [C_out, 1, 3, 3], access [c, 0, kh, kw]
    # Weight is stored in row-major: weight[c, 0, kh, kw] = weight_ptr[c*K*K + kh*K + kw]
    
    for kh in range(K):
        for kw in range(K):
            w_ptr = weight_ptr + c * K * K + kh * K + kw
            w = tl.load(w_ptr)
            
            for h_out in range(H_out):
                for w_out in range(W_out):
                    # Calculate input coordinates with padding (padding=1)
                    h_in = h_out * stride_h + kh - 1
                    w_in = w_out * stride_w + kw - 1
                    
                    # Check bounds
                    if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                        # Input is stored as [B, C, H, W]
                        in_ptr = input_ptr + b * C_out * H * W + c * H * W + h_in * W + w_in
                        x = tl.load(in_ptr)
                        out_val = x * w
                        
                        # Store to output: output[b, c, h_out, w_out]
                        out_ptr = output_ptr + pid * n_spatial + h_out * W_out + w_out
                        tl.store(out_ptr, out_val)
                        
                        mean_acc = mean_acc + out_val
    
    # Store mean (sum / num_elements)
    # num_elements = H_out * W_out
    num_elements = H_out * W_out
    mean_val = mean_acc / tl.cast(num_elements, tl.float32)
    mean_ptr = mean_ptr + pid
    tl.store(mean_ptr, mean_val)


def run_fused_conv2d_mean(input_tensor, weight_tensor, stride_h, stride_w):
    """
    Fused Conv2D + Mean using Triton kernel.
    """
    B, C_in, H, W = input_tensor.shape
    C_out, _, K, _ = weight_tensor.shape
    
    # Calculate output spatial dimensions
    H_out = (H + 2 - K) // stride_h + 1
    W_out = (W + 2 - K) // stride_w + 1
    
    # Allocate outputs
    conv_output = torch.empty((B, C_out, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)
    mean_output = torch.empty((B, C_out), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Grid: B * C_out programs
    grid = (B * C_out,)
    
    # Launch kernel
    fused_conv2d_mean_kernel_impl[grid](
        input_tensor, weight_tensor,
        conv_output, mean_output,
        B, C_out, H, W,
        stride_h, stride_w,
        H_out, W_out, K,
    )
    
    # Reshape mean to [B, C_out, 1, 1]
    mean_output = mean_output.view(B, C_out, 1, 1)
    
    return conv_output, mean_output