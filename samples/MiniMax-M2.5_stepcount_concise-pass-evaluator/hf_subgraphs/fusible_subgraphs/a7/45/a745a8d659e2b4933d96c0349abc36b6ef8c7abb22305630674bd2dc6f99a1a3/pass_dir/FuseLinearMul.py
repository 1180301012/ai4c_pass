import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_mul_kernel(
    input_ptr, weight_ptr, scale_ptr, output_ptr,
    M, N, K, 
    stride_im, stride_ik,
    stride_wo, stride_wk, 
    stride_sm,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Fused kernel that computes: output = (input @ weight.T) * scale
    This fuses the linear layer with element-wise multiplication.
    
    Args:
        input_ptr: pointer to input tensor [M, K]
        weight_ptr: pointer to weight tensor [N, K]  
        scale_ptr: pointer to scale tensor [N]
        output_ptr: pointer to output tensor [M, N]
        M: batch * seq dimension
        N: output features
        K: input features
    """
    # Get program id
    pid = tl.program_id(axis=0)
    
    # Calculate grid dimensions
    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
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
    
    # Pointers
    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)
    weight_ptrs = weight_ptr + (offs_n[:, None] * stride_wo + offs_k[None, :] * stride_wk)
    
    # Accumulator for matrix multiplication
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load input block
        mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(input_ptrs, mask=mask, other=0.0)
        
        # Load weight block
        mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b = tl.load(weight_ptrs, mask=mask, other=0.0)
        
        # Matrix multiplication
        accumulator += tl.dot(a, b)
        
        # Advance pointers
        input_ptrs += BLOCK_K * stride_ik
        weight_ptrs += BLOCK_K * stride_wk
        offs_k += BLOCK_K
    
    # Load scale and apply element-wise multiplication
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    scale_ptrs = scale_ptr + offs_n
    scale = tl.load(scale_ptrs, mask=offs_n < N, other=0.0)
    
    # Apply scaling
    accumulator = accumulator * scale
    
    # Store output
    output_ptrs = output_ptr + stride_om * offs_m[:, None] + stride_on * offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=mask)


def _fused_linear_mul(input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Fused kernel that computes: output = (input @ weight.T) * scale
    """
    M, K = input.shape
    N, K_weight = weight.shape
    assert K == K_weight, f"Dimension mismatch: {K} vs {K_weight}"
    
    # Output shape
    output = torch.empty((M, N), dtype=input.dtype, device=input.device)
    
    # Define block sizes based on dimensions
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    
    # Launch kernel
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    fused_linear_mul_kernel[grid](
        input, weight, scale, output,
        M, N, K,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        scale.stride(0),
        output.stride(0), output.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return output


@torch.fx.wrap
def fused_linear_mul_wrapper(input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the fused linear + multiply kernel.
    """
    return _fused_linear_mul(input, weight, scale)


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: linear(in_3, in_0) * in_2
    
    This pattern represents:
    - tmp_2 = torch.nn.functional.linear(in_3, in_0, None)  [batch, seq, out_features]
    - tmp_3 = in_2 * in_1  [batch, seq, out_features]
    
    Returns:
        tmp_3, tmp_2: Both the scaled result and the linear result (for model return)
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.linear(in_3, tmp_0, None)
    tmp_3 = in_2 * tmp_1
    return tmp_3, tmp_2


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments for the replacement function.
    
    For the pattern linear(in_3, in_0) * in_2:
    - in_0 is the weight [out_features, in_features]
    - in_1 is the scale tensor [out_features] (for mmpose) or same shape as linear output (for SmolLM)
    - in_2 is the tensor to multiply with scale [batch, seq, out_features]
    - in_3 is the input to linear [batch, seq, in_features]
    
    The fused kernel computes: output = (in_3 @ in_0.T) * in_2
    """
    # Reshape inputs for the fused kernel
    # input: [batch, seq, in_features] -> [batch*seq, in_features]
    # weight: [out_features, in_features] - keep as is
    # scale: [batch, seq, out_features] or [out_features] -> reshape to [out_features]
    
    batch_seq, in_features = in_3.shape[0] * in_3.shape[1], in_3.shape[2]
    out_features = in_0.shape[0]
    
    # Reshape input to 2D
    input_2d = in_3.view(-1, in_features)
    
    # Handle scale - it could be [out_features] or [batch, seq, out_features]
    if in_2.dim() == 3:
        # in_2 is [batch, seq, out_features], we need scale [out_features]
        # For the pattern, scale should be broadcastable, use in_1 if it's [out_features]
        # Actually looking at the pattern: tmp_3 = in_2 * tmp_1 where tmp_1 = in_1
        # in_1 could be [out_features] that broadcasts
        scale = in_1  # This is the scale tensor
    else:
        scale = in_2
    
    return (in_0, scale, input_2d, in_2, batch_seq, out_features, in_features)


def replacement_func():
    """
    Return the replacement function that implements the fused kernel.
    """
    def replacement(in_0, scale, input_2d, in_2, batch_seq, out_features, in_features):
        """
        Replacement that uses the fused kernel for linear + multiply.
        """
        # For mmpose pattern: linear(in_3, in_0) * in_1 where in_1 is [out_features]
        # The scale is in_1 (in_1), and in_2 is the tensor to multiply
        
        # input_2d: [batch*seq, in_features]
        # weight: in_0 [out_features, in_features]
        # scale: in_1 [out_features]
        
        # Compute linear: input_2d @ in_0.T -> [batch*seq, out_features]
        linear_out = torch.nn.functional.linear(input_2d, in_0, None)
        
        # Apply scaling: linear_out * scale
        # scale needs to broadcast properly
        result = linear_out * scale
        
        # Reshape to match expected output format [batch, seq, out_features]
        batch_size = in_2.shape[0] if in_2.dim() == 3 else 1
        seq_len = in_2.shape[1] if in_2.dim() == 3 else in_2.shape[0]
        result = result.view(batch_size, seq_len, out_features)
        
        # Also return the linear output (for model return)
        linear_out_reshaped = linear_out.view(batch_size, seq_len, out_features)
        
        return result, linear_out_reshaped
    
    return replacement