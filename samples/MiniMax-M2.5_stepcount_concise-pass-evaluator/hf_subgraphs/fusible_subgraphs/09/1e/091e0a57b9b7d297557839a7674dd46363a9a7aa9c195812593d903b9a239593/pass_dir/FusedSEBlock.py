import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=1),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=1),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1024, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 2048, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def fused_se_block_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, out_ptr,
    stride_in_0, stride_in_1, stride_in_2,
    stride_out,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program processes one channel
    pid = tl.program_id(0)
    
    # Calculate offsets for this channel
    in_0_offset = pid * stride_in_0
    in_1_offset = pid * stride_in_1
    in_2_offset = pid * stride_in_2
    out_offset = pid
    
    # Load in_2 (sigmoid input) - shape [1, 1, 2048]
    in_2_val = tl.load(in_2_ptr + in_2_offset)
    
    # Compute sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-in_2_val))
    
    # Base offsets for in_1 and in_0
    in_1_base = in_1_offset
    in_0_base = in_0_offset
    
    # Accumulator for avg pool
    avg_sum = 0.0
    
    # Process spatial dimension
    for k in range(0, K, BLOCK_SIZE_K):
        k_offset = k + tl.arange(0, BLOCK_SIZE_K)
        mask = k_offset < K
        
        # Load in_1 spatial values
        in_1_offsets = in_1_base + k_offset * stride_in_1
        in_1_vals = tl.load(in_1_ptr + in_1_offsets, mask=mask, other=0.0)
        
        # Load in_0 (residual) spatial values
        in_0_offsets = in_0_base + k_offset * stride_in_0
        in_0_vals = tl.load(in_0_ptr + in_0_offsets, mask=mask, other=0.0)
        
        # Compute attention: sigmoid * in_1
        attention = sigmoid_val * in_1_vals
        
        # Compute add + relu
        sum_vals = attention + in_0_vals
        relu_vals = tl.where(sum_vals > 0, sum_vals, 0.0)
        
        # Accumulate for average pooling
        avg_sum += tl.sum(relu_vals, axis=0)
    
    # Compute average
    avg_result = avg_sum / K
    
    # Store final output
    tl.store(out_ptr + out_offset, avg_result)


def fused_se_block_wrapper(in_0, in_1, in_2):
    """
    Fused SE Block kernel that performs:
    1. sigmoid(in_2) 
    2. view -> expand_as -> multiply with in_1
    3. add in_0 (residual)
    4. relu
    5. adaptive_avg_pool2d
    6. flatten
    """
    batch, channels, H, W = in_1.shape
    spatial = H * W
    
    # Allocate output
    out = torch.empty((1, channels), dtype=torch.float32, device=in_1.device)
    
    # Launch kernel - one program per channel
    grid = (channels,)
    
    fused_se_block_kernel[grid](
        in_0, in_1, in_2, out,
        in_0.stride(0), in_1.stride(0), in_2.stride(0),
        out.stride(0),
        1, channels, spatial,
    )
    
    return out


@torch.fx.wrap
def fused_se_block_kernel_wrapper(in_0, in_1, in_2):
    return fused_se_block_wrapper(in_0, in_1, in_2)


def pattern(in_0, in_1, in_2):
    """Match a minimal SE block computation pattern."""
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    tmp_3 = tmp_3 + in_0
    tmp_5 = torch.nn.functional.relu(tmp_3)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_se_block_kernel_wrapper