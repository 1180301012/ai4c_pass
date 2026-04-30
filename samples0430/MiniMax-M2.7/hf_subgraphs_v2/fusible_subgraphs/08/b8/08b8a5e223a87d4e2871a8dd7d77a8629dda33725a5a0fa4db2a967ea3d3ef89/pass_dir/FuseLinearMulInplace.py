import torch
import triton
import triton.language as tl

@triton.jit
def triton_linear_mul_fused_kernel(
    weight_ptr, input_ptr, mul_input_ptr,
    out_ptr,
    M, N, K,  # batch_size * seq_len, output_dim, input_dim
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for:
    1. Linear: linear_out = input @ weight.T  (where weight is [K, N] -> output is [M, N])
    2. Mul: out = mul_input * linear_out
    
    The mul_input is [M, N] and linear_out is [M, N], element-wise multiply.
    
    This kernel fuses both operations to reduce memory accesses.
    """
    # Program IDs for different work items
    pid = tl.program_id(0)
    
    # Calculate which output element this program handles
    total_outputs = M * N
    if pid >= total_outputs:
        return
    
    # Calculate indices
    batch_idx = pid // N
    col_idx = pid % N
    
    # Accumulator for linear
    acc = 0.0
    
    # Compute linear: input @ weight.T
    # input shape: [M, K], weight shape: [K, N]
    # output shape: [M, N]
    for k in range(0, K, BLOCK_SIZE):
        k_offsets = k + tl.arange(0, BLOCK_SIZE)
        mask_k = k_offsets < K
        
        # Load weight: weight[k, col_idx]
        w_offsets = k_offsets * N + col_idx
        w = tl.load(weight_ptr + w_offsets, mask=mask_k, other=0.0)
        
        # Load input: input[batch_idx, k]
        input_offsets = batch_idx * K + k_offsets
        x = tl.load(input_ptr + input_offsets, mask=mask_k, other=0.0)
        
        acc += tl.sum(x * w)
    
    # Compute element-wise multiply: mul_input * linear_out
    mul_val = tl.load(mul_input_ptr + pid)
    result = mul_val * acc
    
    # Store result
    tl.store(out_ptr + pid, result)


@torch.fx.wrap
def triton_linear_mul_inplace(weight, input_tensor, mul_input):
    """
    Fused linear + element-wise multiply.
    
    Linear: linear_out = input_tensor @ weight.T
    Mul: result = mul_input * linear_out
    
    Returns: result
    """
    # weight: [K, N], input_tensor: [B, S, K], mul_input: [B, S, N], output: [B, S, N]
    
    B, S, K = input_tensor.shape
    N = weight.shape[0]
    M = B * S
    
    # Flatten for linear computation
    input_flat = input_tensor.view(M, K)
    mul_input_flat = mul_input.view(M, N)
    
    # Allocate output
    output = torch.empty((B, S, N), dtype=input_tensor.dtype, device=input_tensor.device)
    output_flat = output.view(M, N)
    
    # Grid configuration
    BLOCK_SIZE = 128
    total_programs = M * N
    
    # Launch kernel
    triton_linear_mul_fused_kernel[(total_programs,)](
        weight_ptr=weight,
        input_ptr=input_flat,
        mul_input_ptr=mul_input_flat,
        out_ptr=output_flat,
        M=M, N=N, K=K,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_0, in_1, in_2):
    """
    Pattern: linear(in_1, in_0, None) * in_2
    
    in_0: weight tensor [K, N]
    in_1: input tensor [B, S, K]
    in_2: mul_input tensor [B, S, N]
    
    Returns: (linear_output * in_2)
    """
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = in_2 * linear
    return (tmp_2,)


def replacement_args(in_0, in_1, in_2):
    # For SmolLM3/gemma pattern: in_0 is weight, in_1 is input, in_2 is mul_input
    # weight: [11008, 2048], input: [B, seq, 2048], mul_input: [B, seq, 11008]
    return (in_0, in_1, in_2)


def replacement_func():
    return triton_linear_mul_inplace