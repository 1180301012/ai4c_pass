import torch
import triton
import triton.language as tl

# Pattern: 1x1 Conv2D operation
# torch.conv2d(input, weight, bias, stride, padding, dilation, groups)
# For 1x1 conv: stride=(1,1), padding=(0,0), dilation=(1,1), groups=1
def pattern(in_2, in_1, in_0):
    """
    Match 1x1 Conv2D pattern with bias.
    This is a pointwise convolution that can be optimized as a batched matmul.
    """
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return tmp_2


def replacement_args(in_2, in_1, in_0):
    # Return the conv inputs: input, weight, bias
    return (in_2, in_1, in_0)


# Optimized Triton kernel for 1x1 Conv2D
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_kernel(
    input_ptr, weight_ptr, output_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_im, stride_ip,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Triton kernel for 1x1 Conv2D optimized as matrix multiplication.
    """
    # Get program id
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, M - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        mask_k = k_offs < K
        
        # Load input block
        input_ptrs = input_ptr + (offs_m[:, None] * stride_im + k_offs[None, :] * stride_ip)
        input_mask = (offs_m[:, None] < M) & (mask_k[None, :])
        input = tl.load(input_ptrs, mask=input_mask, other=0.0)
        
        # Load weight block
        weight_ptrs = weight_ptr + (k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        weight_mask = (mask_k[:, None] & (offs_n[None, :] < N))
        weight = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        
        # Matrix multiply
        accumulator += tl.dot(input, weight)
    
    # Store output
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_om * offs_m[:, None] + stride_on * offs_n[None, :]
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=output_mask)


@torch.fx.wrap
def conv1x1_wrapper(input_tensor, weight_tensor, bias_tensor):
    """
    1x1 Conv2D implemented as optimized matrix multiplication using Triton.
    
    Args:
        input_tensor: (batch, in_channels, height, width)
        weight_tensor: (out_channels, in_channels, 1, 1)
        bias_tensor: (out_channels,)
    
    Returns:
        output_tensor: (batch, out_channels, height, width)
    """
    batch, in_channels, height, width = input_tensor.shape
    out_channels = weight_tensor.shape[0]
    
    # Reshape for matrix multiplication:
    # Input: (batch, in_channels, H, W) -> (batch*H*W, in_channels)
    # Weight: (out_channels, in_channels, 1, 1) -> (out_channels, in_channels)
    M = batch * height * width  # Total spatial positions
    K = in_channels  # Input channels
    N = out_channels  # Output channels
    
    # Reshape input to 2D (M, K)
    input_reshaped = input_tensor.permute(0, 2, 3, 1).reshape(M, K).contiguous()
    
    # Reshape weight to 2D (N, K) - no need to squeeze since it's already (out, in, 1, 1)
    weight_reshaped = weight_tensor.squeeze(-1).squeeze(-1).contiguous()  # (out_channels, in_channels)
    
    # Allocate output
    output_reshaped = torch.empty((M, N), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel - grid calculates how many blocks needed
    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 256),)
    
    conv1x1_kernel[grid](
        input_ptr=input_reshaped,
        weight_ptr=weight_reshaped,
        output_ptr=output_reshaped,
        M=M, N=N, K=K,
        stride_im=input_reshaped.stride(0),
        stride_ip=input_reshaped.stride(1),
        stride_wn=weight_reshaped.stride(1),
        stride_wk=weight_reshaped.stride(0),
        stride_om=output_reshaped.stride(0),
        stride_on=output_reshaped.stride(1),
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=64,
    )
    
    # Add bias
    if bias_tensor is not None:
        output_reshaped = output_reshaped + bias_tensor
    
    # Reshape output back to (batch, out_channels, height, width)
    output = output_reshaped.reshape(batch, height, width, out_channels)
    output = output.permute(0, 3, 1, 2).contiguous()
    
    return output


def replacement_func():
    return conv1x1_wrapper