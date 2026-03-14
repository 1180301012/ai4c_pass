import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Match pattern: linear -> dropout(p=0.1, training=False) -> add
    For BERT graphs: dropout has p=0.1 and training=False (inference mode)
    
    BERT model:
        tmp_2 = torch.nn.functional.linear(in_3, tmp_1, tmp_0)  # input=in_3, weight=in_1, bias=in_0
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)  # p=0.1, training=False
        tmp_4 = tmp_3 + in_2  # in_2 is residual (embeddings)
        return (tmp_4,)
    
    Returns only add_out (the tuple has single element)
    """
    # Linear: in_3 @ in_1.T + in_0
    linear_out = torch.nn.functional.linear(in_3, in_1, in_0)
    
    # Dropout with p=0.1, training=False (4 args: input, p, training, inplace)
    dropout_out = torch.nn.functional.dropout(linear_out, 0.1, False, False)
    
    # Add residual (in_2)
    add_out = dropout_out + in_2
    
    return add_out


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments: bias, weight, residual, input
    """
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_linear_add_kernel_bert(
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
    Fused kernel: Linear + Add (dropout is identity when training=False)
    
    Computes: output = input @ weight.T + bias + residual
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
    
    # Pointers
    input_ptrs = input_ptr + (offs_m[:, None] * input_stride0 + offs_k[None, :] * input_stride1)
    weight_ptrs = weight_ptr + (offs_n[:, None] * weight_stride0 + offs_k[None, :] * weight_stride1)
    residual_ptrs = residual_ptr + (offs_m[:, None] * residual_stride0 + offs_n[None, :] * residual_stride1)
    output_ptrs = output_ptr + (offs_m[:, None] * output_stride0 + offs_n[None, :] * output_stride1)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main matmul loop
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        input_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(input_ptrs, mask=input_mask, other=0.0)
        
        weight_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        
        accumulator += tl.dot(a, b)
        
        input_ptrs += BLOCK_K * input_stride1
        weight_ptrs += BLOCK_K * weight_stride1
        offs_k += BLOCK_K
    
    # Add bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias
    
    # Add residual (dropout is identity when training=False)
    residual_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    residual = tl.load(residual_ptrs, mask=residual_mask, other=0.0)
    output = accumulator + residual
    
    # Store
    tl.store(output_ptrs, output, mask=residual_mask)


@torch.fx.wrap
def fused_kernel_wrapper_bert(bias, weight, residual, input_tensor):
    """
    Wrapper for fused linear + dropout + add kernel for BERT
    """
    # Get shapes
    original_shape = input_tensor.shape
    n_dims = len(original_shape)
    
    # Flatten all leading dims except last
    M = 1
    for dim in original_shape[:-1]:
        M *= dim
    K = original_shape[-1]
    N = weight.shape[0]
    
    # Handle multi-dimensional input
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
    
    # Launch kernel - use 2D grid
    grid = (triton.cdiv(M, 32) * triton.cdiv(N, 32),)
    
    # Strides
    input_stride0, input_stride1 = input_tensor.stride()
    weight_stride0, weight_stride1 = weight.stride()
    residual_stride0, residual_stride1 = residual.stride()
    output_stride0, output_stride1 = output.stride()
    
    fused_linear_add_kernel_bert[grid](
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
    
    return output


def replacement_func():
    return fused_kernel_wrapper_bert