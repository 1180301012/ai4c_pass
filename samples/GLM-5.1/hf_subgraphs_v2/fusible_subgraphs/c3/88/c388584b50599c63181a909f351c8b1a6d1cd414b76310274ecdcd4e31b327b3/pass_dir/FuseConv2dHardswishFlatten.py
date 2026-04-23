import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(conv2d, True)
    tmp_4 = tmp_3.flatten(1, -1)
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_conv1x1_hardswish_flatten_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_input_batch, stride_input_channel,
    stride_weight_out, stride_weight_in,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused kernel for 1x1 conv2d + hardswish + flatten.
    
    Works directly with 4D tensor strides without reshaping.
    For 1x1 conv with H=W=1:
    - Input: [B, C_in, 1, 1] accessed as 2D [B, C_in] using stride_input_batch, stride_input_channel
    - Weight: [C_out, C_in, 1, 1] accessed as 2D [C_out, C_in] using stride_weight_out, stride_weight_in
    - Bias: [C_out]
    - Output: [B, C_out] (the flattened result)
    
    M = B (batch size), N = C_out (output channels), K = C_in (input channels)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator for matmul
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Matmul: accumulate over K (input channels) dimension
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        
        # Load input tile: [BLOCK_M, BLOCK_K]
        # Accessing [B, C_in, 1, 1] as if it's [B, C_in] using strides
        input_ptrs = input_ptr + offs_m[:, None] * stride_input_batch + k_offs[None, :] * stride_input_channel
        input_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        input_tile = tl.load(input_ptrs, mask=input_mask, other=0.0)
        
        # Load weight tile: [BLOCK_K, BLOCK_N]
        # Accessing [C_out, C_in, 1, 1] as if it's [C_out, C_in] using strides
        # For matmul we need [K, N] = [C_in, C_out], so we transpose the indexing
        weight_ptrs = weight_ptr + k_offs[:, None] * stride_weight_in + offs_n[None, :] * stride_weight_out
        weight_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        weight_tile = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        
        # Accumulate: input @ weight (transposed)
        acc += tl.dot(input_tile, weight_tile, allow_tf32=True)
    
    # Add bias
    bias_ptrs = bias_ptr + offs_n
    bias_mask = offs_n < N
    bias_vals = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    acc += bias_vals[None, :]
    
    # Apply hardswish: x * relu6(x + 3) / 6
    # relu6(x) = min(max(x, 0), 6)
    # hardswish(x) = x * relu6(x + 3) / 6
    
    # Step 1: x + 3
    val_plus_3 = acc + 3.0
    
    # Step 2: relu6(x + 3) = min(max(x + 3, 0), 6)
    relu6_result = tl.minimum(tl.maximum(val_plus_3, 0.0), 6.0)
    
    # Step 3: x * relu6(x + 3)
    hardswish_result = acc * relu6_result
    
    # Step 4: divide by 6
    result = hardswish_result / 6.0
    
    # Store output [B, C_out]
    output_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, result, mask=output_mask)


@torch.fx.wrap
def fused_conv1x1_hardswish_flatten(bias, weight, input_tensor):
    """
    Fused 1x1 conv2d + hardswish + flatten implementation.
    
    Works directly with 4D tensors using strides, avoiding reshape/view operations
    (which are not permitted in the replacement function).
    
    For 1x1 conv with H=W=1:
    - Conv2d is equivalent to: output = input @ weight.T + bias
    - Input [B, C_in, 1, 1] is accessed as [B, C_in] via strides
    - Weight [C_out, C_in, 1, 1] is accessed as [C_out, C_in] via strides
    - Output is [B, C_out] (the result of flatten(1, -1))
    """
    B = input_tensor.shape[0]
    C_in = input_tensor.shape[1]
    C_out = weight.shape[0]
    
    M = B
    N = C_out
    K = C_in
    
    # Create output tensor directly in the final shape [B, C_out]
    # This avoids any reshape/view operations
    output = torch.empty((B, C_out), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Use strides from original 4D tensors to access data as 2D
    # Input [B, C_in, 1, 1]: stride(0) is batch stride, stride(1) is channel stride
    stride_input_batch = input_tensor.stride(0)
    stride_input_channel = input_tensor.stride(1)
    
    # Weight [C_out, C_in, 1, 1]: stride(0) is output channel stride, stride(1) is input channel stride
    stride_weight_out = weight.stride(0)
    stride_weight_in = weight.stride(1)
    
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 32
    
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    grid = (grid_m, grid_n)
    
    fused_conv1x1_hardswish_flatten_kernel[grid](
        input_tensor, weight, bias, output,
        M, N, K,
        stride_input_batch, stride_input_channel,
        stride_weight_out, stride_weight_in,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return (output,)


def replacement_func():
    return fused_conv1x1_hardswish_flatten