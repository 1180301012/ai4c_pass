import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Match pattern: linear -> dropout(p=0.1, training=False) -> add
    
    BERT model pattern:
        tmp_2 = torch.nn.functional.linear(in_3, tmp_1, tmp_0)  # input=in_3, weight=in_1, bias=in_0
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)  # p=0.1, training=False
        tmp_4 = tmp_3 + in_2  # in_2 is residual
        return (tmp_4,)
    """
    # Linear: in_3 @ in_1.T + in_0
    linear_out = torch.nn.functional.linear(in_3, in_1, in_0)
    
    # Dropout with p=0.1, training=False (positional args)
    dropout_out = torch.nn.functional.dropout(linear_out, 0.1, False, False)
    
    # Add residual (in_2)
    add_out = dropout_out + in_2
    
    return add_out


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments: bias, weight, input, residual
    """
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_linear_add_kernel(
    # Pointers
    input_ptr, weight_ptr, bias_ptr, residual_ptr, output_ptr,
    # Shapes
    M, N, K,
    # Strides
    input_stride0, input_stride1,
    weight_stride0, weight_stride1,
    residual_stride0, residual_stride1,
    output_stride0, output_stride1,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused kernel: Linear + Add
    
    Computes: output = input @ weight.T + bias + residual
    PyTorch linear does: input @ weight.T + bias
    """
    # Get program IDs for 2D tiling
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
    
    # Input pointer: [M, K] - access as input[m, k]
    input_ptrs = input_ptr + (offs_m[:, None] * input_stride0 + offs_k[None, :] * input_stride1)
    
    # Weight pointer: weight is [N, K], but we need weight.T which is [K, N]
    # So we access weight.T[k, n] = weight[n, k]
    # weight.T[k, n] means: for fixed n (output feature), iterate over k (input feature)
    # So pointer is: weight_ptr + k * weight_stride1 + n * weight_stride0
    weight_ptrs = weight_ptr + (offs_k[:, None] * weight_stride1 + offs_n[None, :] * weight_stride0)
    
    # Residual pointer: [M, N]
    residual_ptrs = residual_ptr + (offs_m[:, None] * residual_stride0 + offs_n[None, :] * residual_stride1)
    # Output pointer: [M, N]
    output_ptrs = output_ptr + (offs_m[:, None] * output_stride0 + offs_n[None, :] * output_stride1)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main matmul loop: input @ weight.T
    # input: [BLOCK_M, BLOCK_K], weight.T: [BLOCK_K, BLOCK_N]
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        input_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(input_ptrs, mask=input_mask, other=0.0)
        
        weight_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        # Load weight.T[k, n] = weight[n, k]
        b = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        
        # Compute dot product - this gives us input @ weight.T
        accumulator += tl.dot(a, b)
        
        input_ptrs += BLOCK_K * input_stride1
        weight_ptrs += BLOCK_K * weight_stride1
        offs_k += BLOCK_K
    
    # Add bias - bias is [N], broadcast to [M, N]
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias
    
    # Add residual
    residual_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    residual = tl.load(residual_ptrs, mask=residual_mask, other=0.0)
    output = accumulator + residual
    
    # Store
    tl.store(output_ptrs, output, mask=residual_mask)


def compute_linear_fused_bert(input_tensor, weight, bias, residual):
    """
    Compute fused linear + add using Triton kernel for BERT pattern
    Returns just the output (add_out), not a tuple
    """
    # Handle multi-dimensional input (e.g., [batch, seq, hidden])
    original_shape = input_tensor.shape
    n_dims = len(original_shape)
    
    # Flatten all leading dims except last: [batch, seq, hidden] -> [batch*seq, hidden]
    M = 1
    for dim in original_shape[:-1]:
        M *= dim
    K = original_shape[-1]
    N = weight.shape[0]
    
    # Reshape input and residual to 2D
    if n_dims > 2:
        input_tensor = input_tensor.reshape(M, K)
        residual = residual.reshape(M, N)
    
    # Ensure same device
    device = input_tensor.device
    weight = weight.to(device)
    bias = bias.to(device) if bias is not None else None
    residual = residual.to(device)
    
    # Allocate output
    output = torch.empty((M, N), device=device, dtype=input_tensor.dtype)
    
    # Launch kernel
    grid = (triton.cdiv(M, 32) * triton.cdiv(N, 32),)
    
    # Strides
    input_stride0, input_stride1 = input_tensor.stride()
    weight_stride0, weight_stride1 = weight.stride()
    residual_stride0, residual_stride1 = residual.stride()
    output_stride0, output_stride1 = output.stride()
    
    fused_linear_add_kernel[grid](
        input_tensor, weight, bias, residual, output,
        M, N, K,
        input_stride0, input_stride1,
        weight_stride0, weight_stride1,
        residual_stride0, residual_stride1,
        output_stride0, output_stride1,
        32, 64, 64,
    )
    
    # Reshape back to original shape (except last dim)
    if n_dims > 2:
        output = output.reshape(*original_shape[:-1], N)
    
    # Return just the output (not a tuple) for BERT pattern
    return output


@torch.fx.wrap
def fused_kernel_wrapper(bias, weight, input_tensor, residual):
    return compute_linear_fused_bert(input_tensor, weight, bias, residual)


def replacement_func():
    return fused_kernel_wrapper