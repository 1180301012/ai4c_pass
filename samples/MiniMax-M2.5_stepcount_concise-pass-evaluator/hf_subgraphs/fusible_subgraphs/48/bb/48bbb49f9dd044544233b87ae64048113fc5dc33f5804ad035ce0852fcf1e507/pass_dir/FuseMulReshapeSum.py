import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match the computation pattern:
    - softmax(in_2, dim=2) -> tmp_2
    - tmp_2.reshape(-1, 17, 64, 64) -> tmp_3
    - tmp_3.mul(in_0) -> tmp_4
    - tmp_4.reshape(B, 17, -1) -> tmp_5
    - torch.sum(tmp_5, dim=2, keepdim=True) -> tmp_6
    - tmp_3.mul(in_1) -> tmp_7
    - tmp_7.reshape(B, 17, -1) -> tmp_8
    - torch.sum(tmp_8, dim=2, keepdim=True) -> tmp_9
    - torch.cat([tmp_6, tmp_9], dim=-1) -> tmp_10
    
    Returns (tmp_3, tmp_10) to match the model's return.
    """
    tmp_2 = torch.nn.functional.softmax(in_2, dim=2)
    tmp_3 = tmp_2.reshape(-1, 17, 64, 64)
    
    tmp_4 = tmp_3.mul(in_0)
    tmp_5 = tmp_4.reshape(-1, 17, -1)
    tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
    
    tmp_7 = tmp_3.mul(in_1)
    tmp_8 = tmp_7.reshape(-1, 17, -1)
    tmp_9 = torch.sum(tmp_8, dim=2, keepdim=True)
    
    tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
    
    return tmp_3, tmp_10


def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for the replacement function."""
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64, 'num_warps': 1}),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128, 'num_warps': 1}),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256, 'num_warps': 2}),
        triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 64, 'num_warps': 1}),
        triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 128, 'num_warps': 2}),
        triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 64, 'num_warps': 2}),
    ],
    key=['B', 'num_keys'],
)
@triton.jit
def fused_mul_reshape_sum_kernel(
    softmax_ptr, in_0_ptr, in_1_ptr,
    out_3_ptr, out_10_ptr,
    B, num_keys, H, W,
    stride_softmax_b, stride_softmax_h, stride_softmax_w,
    stride_in0_b, stride_in0_h, stride_in0_w,
    stride_in1_b, stride_in1_h, stride_in1_w,
    stride_out3_b, stride_out3_h, stride_out3_w,
    stride_out10_b, stride_out10_h, stride_out10_w,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """Fused kernel that performs: mul + reshape + sum in a single kernel."""
    
    # Block index for output batch and key dimension
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    # Calculate output pointer offsets
    # Output 3: [B, 17, 64, 64]
    off_out3 = pid_b * stride_out3_b + pid_k * stride_out3_h
    
    # Output 10: [B, 17, 1, 2] (concatenated sums from in_0 and in_1)
    off_out10 = pid_b * stride_out10_b + pid_k * stride_out10_h
    
    # Load in_0 and in_1 values for this key
    # in_0 shape: [1, 1, 1, 64] -> broadcast to [B, 17, 64, 64] 
    # in_0 indexed as [0, 0, pid_k, :]
    in_0_val = tl.load(in_0_ptr + pid_k * stride_in0_w + tl.arange(0, BLOCK_SIZE_N))
    
    # in_1 shape: [1, 1, 64, 1] -> broadcast to [B, 17, 64, 64]
    # in_1 indexed as [0, 0, :, pid_k]
    in_1_val = tl.load(in_1_ptr + pid_k * stride_in1_h + tl.arange(0, BLOCK_SIZE_N))
    
    # Initialize accumulators for sum along W dimension (64)
    sum_out0 = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    sum_out1 = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Process each h dimension (17 keypoints)
    for h in range(H):
        # Load softmax data for this batch, key, h
        # softmax shape: [B, 17, 4096] where 4096 = 64*64
        # We need to load W elements
        off_softmax = pid_b * stride_softmax_b + h * stride_softmax_h + pid_k * W * stride_softmax_w
        
        # Load the 64x64 block
        softmax_vals = tl.load(softmax_ptr + off_softmax + tl.arange(0, BLOCK_SIZE_N) * stride_softmax_w)
        
        # Store to output_3 (reshaped softmax)
        off_out3_h = off_out3 + h * stride_out3_h
        tl.store(out_3_ptr + off_out3_h + tl.arange(0, BLOCK_SIZE_N), softmax_vals)
        
        # Multiply with in_0 and accumulate
        mul_out0 = softmax_vals * in_0_val
        sum_out0 += mul_out0
        
        # Multiply with in_1 and accumulate
        mul_out1 = softmax_vals * in_1_val
        sum_out1 += mul_out1
    
    # Store the sum results to output_10
    # Shape: [B, 17, 1, 2] - last dim is concatenation of sums
    tl.store(out_10_ptr + off_out10, sum_out0)
    tl.store(out_10_ptr + off_out10 + stride_out10_w, sum_out1)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2):
    """
    Wrapper function that launches the fused Triton kernel.
    This fuses: softmax + reshape + mul + sum operations.
    """
    # Compute softmax
    softmax = torch.nn.functional.softmax(in_2, dim=2)
    
    # Get dimensions
    B = softmax.shape[0]  # Batch size
    H = 17  # Number of keypoints
    W = 64  # Spatial dimension (64*64 = 4096)
    
    # Output 3: reshaped softmax [B, 17, 64, 64]
    out_3 = torch.empty((B, H, W, W), dtype=softmax.dtype, device=softmax.device)
    
    # Output 10: [B, 17, 1, 2] - concatenated sums
    out_10 = torch.empty((B, H, 1, 2), dtype=softmax.dtype, device=softmax.device)
    
    # Launch kernel
    # Grid: (B, H) - each block handles one batch element and one keypoint
    grid = (B, H)
    
    fused_mul_reshape_sum_kernel[grid](
        softmax, in_0, in_1,
        out_3, out_10,
        B, H, W,
        softmax.stride(0), softmax.stride(1), softmax.stride(2),
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        out_3.stride(0), out_3.stride(1), out_3.stride(2),
        out_10.stride(0), out_10.stride(1), out_10.stride(2),
    )
    
    return out_3, out_10


def replacement_func():
    """Return the replacement function."""
    return fused_kernel_wrapper