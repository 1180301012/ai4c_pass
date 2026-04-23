import torch
import triton
import triton.language as tl

# Pattern matching function - matches softmax -> mul -> sum pattern
def pattern(in_0, in_1):
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1, "FusedSoftmaxWeightedSum")

# Autotuning configuration
@triton.autotune(
    configs=[
        triton.Config({}, num_stages=3, num_warps=4),
        triton.Config({}, num_stages=3, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=8),
    ],
    key=['batch_size', 'channels', 'height', 'width'],
)
@triton.jit
def fused_softmax_weighted_sum_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    batch_size, num_classes, channels, height, width,
    stride_in_0_b, stride_in_0_class, stride_in_0_c, stride_in_0_h, stride_in_0_w,
    stride_in_1_b, stride_in_1_class, stride_in_1_c, stride_in_1_h, stride_in_1_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w
):
    # Get program id for parallelization over batch, channels, and spatial positions
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    pid_h = pid_hw // width
    pid_w = pid_hw % width
    
    # Compute offsets for in_0: [B, 2, C, H, W]
    offs_in_0 = (pid_b * stride_in_0_b + 
                 tl.arange(0, 2) * stride_in_0_class +  # classes - use stride[1]
                 pid_c * stride_in_0_c + 
                 pid_h * stride_in_0_h + 
                 pid_w * stride_in_0_w)
    
    # Compute offsets for in_1: [B, 2, C, 1, 1]
    offs_in_1 = (pid_b * stride_in_1_b + 
                 tl.arange(0, 2) * stride_in_1_class +  # classes - use stride[1]
                 pid_c * stride_in_1_c)
    
    # Load in_1 values [B, 2, C, 1, 1] - these are the same across H,W
    in_1_vals = tl.load(in_1_ptr + offs_in_1)
    
    # Compute softmax over dim=1 (the class dimension with size 2)
    # Softmax: exp(x) / sum(exp(x))
    # Since in_1 has shape [B, 2, C, 1, 1], we need to softmax over dim=1
    exp_vals = tl.exp(in_1_vals - tl.max(in_1_vals, axis=0))
    softmax_vals = exp_vals / tl.sum(exp_vals, axis=0)
    
    # Load in_0 values [B, 2, C, H, W] for this position
    in_0_vals = tl.load(in_0_ptr + offs_in_0)
    
    # Multiply and sum over dim=1 (class dimension)
    # weighted_sum = sum(softmax_vals * in_0_vals)
    weighted_sum = tl.sum(softmax_vals * in_0_vals, axis=0)
    
    # Store result to output [B, C, H, W]
    offs_out = (pid_b * stride_out_b + 
                pid_c * stride_out_c + 
                pid_h * stride_out_h + 
                pid_w * stride_out_w)
    tl.store(out_ptr + offs_out, weighted_sum)

@torch.fx.wrap
def triton_fused_softmax_weighted_sum_dispatcher(in_0, in_1, route="FusedSoftmaxWeightedSum"):
    """
    Shared dispatcher that routes to the appropriate kernel based on route string.
    When output_pass_replacement_func_limit=1, all passes share this same replacement_func().
    """
    if route != "FusedSoftmaxWeightedSum":
        raise NotImplementedError(f"Unknown route: {route}")
    
    # Get tensor properties
    # in_0: [B, 2, C, H, W], in_1: [B, 2, C, 1, 1]
    B, num_classes, C, H, W = in_0.shape
    dtype = in_0.dtype
    device = in_0.device
    
    # Allocate output tensor [B, C, H, W]
    out = torch.empty((B, C, H, W), dtype=dtype, device=device)
    
    # Calculate strides (5D tensors)
    # in_0.stride() returns (stride_b, stride_class, stride_c, stride_h, stride_w)
    # in_1.stride() returns (stride_b, stride_class, stride_c, stride_h, stride_w)
    # out.stride() returns (stride_b, stride_c, stride_h, stride_w)
    stride_in_0 = in_0.stride()
    stride_in_1 = in_1.stride()
    stride_out = out.stride()
    
    # Grid dimensions
    # Grid X: batch dimension
    # Grid Y: channel dimension
    # Grid Z: spatial dimension (H * W)
    grid = (B, C, H * W)
    
    # Launch kernel with all strides
    fused_softmax_weighted_sum_kernel[grid](
        in_0, in_1, out,
        B, num_classes, C, H, W,
        stride_in_0[0], stride_in_0[1], stride_in_0[2], stride_in_0[3], stride_in_0[4],  # in_0 strides
        stride_in_1[0], stride_in_1[1], stride_in_1[2], stride_in_1[3], stride_in_1[4],  # in_1 strides
        stride_out[0], stride_out[1], stride_out[2], stride_out[3]  # out strides
    )
    
    return out

def replacement_func():
    return triton_fused_softmax_weighted_sum_dispatcher