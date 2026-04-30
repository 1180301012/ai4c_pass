import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.silu(conv2d, inplace=False)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_conv1x1_silu_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bias,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused 1x1 Conv2D + SiLU kernel.
    Computes: C = SiLU(A @ B^T + bias)
    where A = [M, K] (input flattened), B = [N, K] (weight), bias = [N]
    SiLU(x) = x * sigmoid(x)
    
    Note: dropout(p=0.0, train=False) is identity, so it's skipped.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Group ordering for better L2 cache utilization
    GROUP_SIZE_M: tl.constexpr = 8
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Initialize accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension (input channels)
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        
        # Load A tile: [BLOCK_M, BLOCK_K] - input data
        a_offsets = offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # Load B tile: [BLOCK_K, BLOCK_N] - weight data (transposed view)
        # B is stored as [N, K], accessed as [K, N]
        b_offsets = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Matrix multiply using tensor cores for fp16/bf16, CUDA cores for fp32
        acc += tl.dot(a, b, allow_tf32=False)
    
    # Load bias: [BLOCK_N]
    bias_offsets = offs_n * stride_bias
    bias = tl.load(bias_ptr + bias_offsets, mask=mask_n, other=0.0)
    acc += bias[None, :].to(tl.float32)
    
    # Apply SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
    silu_out = acc * tl.sigmoid(acc)
    
    # Store output tile
    c_offsets = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptr + c_offsets, silu_out, mask=c_mask)


@torch.fx.wrap
def fused_conv1x1_silu(bias, weight, input_tensor):
    """
    Fused 1x1 Conv2D + SiLU + identity dropout.
    For 1x1 convolution, the operation is equivalent to:
    output = SiLU(input_flat @ weight.T + bias)
    then reshape back to 4D.
    """
    batch, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Reshape input to 2D: [batch*height*width, in_channels]
    a = input_tensor.reshape(batch * height * width, in_channels)
    # Reshape weight to 2D: [out_channels, in_channels] (1x1 conv)
    b = weight.reshape(out_channels, in_channels)
    
    M = batch * height * width
    K = in_channels
    N = out_channels
    
    # Create output tensor (2D)
    c = torch.empty((M, N), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Grid: total number of tiles
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    fused_conv1x1_silu_kernel[grid](
        a, b, c, bias,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(1), b.stride(0),  # transposed access: stride_bk=stride(1), stride_bn=stride(0)
        c.stride(0), c.stride(1),
        bias.stride(0),
    )
    
    # Reshape output back to 4D: [batch, out_channels, height, width]
    output = c.reshape(batch, out_channels, height, width)
    return output


def replacement_func():
    return fused_conv1x1_silu