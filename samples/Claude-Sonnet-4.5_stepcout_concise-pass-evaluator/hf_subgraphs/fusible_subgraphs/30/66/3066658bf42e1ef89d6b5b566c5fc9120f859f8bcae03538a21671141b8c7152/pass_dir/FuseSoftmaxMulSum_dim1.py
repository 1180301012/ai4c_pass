import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """Match the pattern: softmax(in_1, dim=1) * in_0, then sum(result, dim=1)
    
    This fuses three operations:
    1. softmax(in_1, dim=1) - compute softmax along dim 1
    2. in_0 * softmax_result - element-wise multiplication (with broadcasting)
    3. sum(multiply_result, dim=1) - sum along dim 1
    """
    tmp_softmax = torch.softmax(in_1, dim=1)
    tmp_mul = in_0 * tmp_softmax
    tmp_sum = torch.sum(tmp_mul, dim=1)
    return tmp_sum


def replacement_args(in_0, in_1):
    """Extract arguments needed for the replacement kernel."""
    return (in_0, in_1)


# Optimized Triton kernel with proper vectorization and memory coalescing
@triton.jit
def fused_softmax_mul_sum_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    stride_in_0_feat, stride_in_0_h, stride_in_0_w,
    stride_in_1_feat,
    stride_out_feat, stride_out_h, stride_out_w,
    FEATURES, H, W,
):
    """Fused kernel that computes: sum(softmax(in_1, dim=1) * in_0, dim=1)
    
    Optimized for memory coalescing by processing in feature-major order.
    
    Args:
        in_0: [1, 2, FEATURES, H, W]
        in_1: [1, 2, FEATURES, 1, 1]
        Output: [1, FEATURES, H, W]
    """
    # Get position
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    
    total_outputs = FEATURES * H * W
    
    # Process multiple outputs for better load balancing
    for idx in range(pid, total_outputs, num_pids):
        # Decode indices - process in feature-major order for memory coalescing
        feat_idx = idx // (H * W)
        rem = idx % (H * W)
        h_idx = rem // W
        w_idx = rem % W
        
        # Compute softmax for all channels at this position
        # Load both channel values
        c0_val = tl.load(in_1_ptr + feat_idx * stride_in_1_feat).to(tl.float32)
        c1_val = tl.load(in_1_ptr + 1 * stride_in_1_feat + feat_idx * stride_in_1_feat).to(tl.float32)
        
        # Compute max for numerical stability
        max_val = c0_val if c0_val > c1_val else c1_val
        
        # Compute exp and sum
        exp0 = tl.exp(c0_val - max_val)
        exp1 = tl.exp(c1_val - max_val)
        exp_sum = exp0 + exp1
        
        # Load in_0 values at this position for both channels
        in0 = tl.load(in_0_ptr + feat_idx * stride_in_0_feat + 
                      h_idx * stride_in_0_h + 
                      w_idx * stride_in_0_w).to(tl.float32)
        in1 = tl.load(in_0_ptr + stride_in_0_feat + 
                      feat_idx * stride_in_0_feat + 
                      h_idx * stride_in_0_h + 
                      w_idx * stride_in_0_w).to(tl.float32)
        
        # Weighted sum
        weighted = in0 * exp0 + in1 * exp1
        
        # Final result = weighted sum / softmax sum
        result = weighted / exp_sum
        
        # Store
        tl.store(out_ptr + feat_idx * stride_out_feat + 
                        h_idx * stride_out_h + 
                        w_idx * stride_out_w, result)


@torch.fx.wrap
def fused_softmax_mul_sum_wrapper(in_0, in_1):
    """Wrapper function that launches the fused Triton kernel."""
    # Get input shape: [1, 2, FEATURES, H, W]
    BATCH, CHANNELS, FEATURES, H, W = in_0.shape
    
    # Output shape: [1, FEATURES, H, W]
    out_shape = (BATCH, FEATURES, H, W)
    
    # Allocate output
    out = torch.empty(out_shape, device=in_0.device, dtype=in_0.dtype)
    
    # Get strides - contiguity matters for memory access
    s0 = in_0.stride()  # [feat*h*w*2, h*w*2, h*w, w, 1] typically
    s1 = in_1.stride()
    so = out.stride()
    
    # Use grid size to match output elements with good parallelism
    total_outputs = FEATURES * H * W
    grid_size = min(256, total_outputs)
    
    # Launch kernel
    fused_softmax_mul_sum_kernel[(grid_size,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        stride_in_0_feat=s0[2], stride_in_0_h=s0[3], stride_in_0_w=s0[4],
        stride_in_1_feat=s1[2],
        stride_out_feat=so[1], stride_out_h=so[2], stride_out_w=so[3],
        FEATURES=FEATURES, H=H, W=W,
    )
    
    return out


def replacement_func():
    """Return the replacement function."""
    return fused_softmax_mul_sum_wrapper