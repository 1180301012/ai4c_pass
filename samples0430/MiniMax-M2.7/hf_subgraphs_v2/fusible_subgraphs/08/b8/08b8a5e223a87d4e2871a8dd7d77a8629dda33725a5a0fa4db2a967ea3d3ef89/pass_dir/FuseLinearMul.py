import torch
import triton
import triton.language as tl

@triton.jit
def triton_linear_mul_kernel(
    weight_ptr, input_ptr, scale_ptr, mul_input_ptr,
    out1_ptr, out2_ptr,
    M, N, K,  # batch_size * seq_len, input_dim, output_dim
    batch_stride, seq_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for:
    1. Linear: output = input @ weight.T  (where weight is [K, N] -> output is [M, N])
    2. Mul: result = mul_input * scale (broadcast)
    
    Both operations are independent and can be computed in parallel.
    """
    # Program IDs for different work items
    pid = tl.program_id(0)
    
    # Calculate which output element this program handles
    # We compute both outputs, so we need M * N total work
    total_outputs = M * N
    if pid >= total_outputs:
        return
    
    # Calculate indices
    batch_idx = pid // N
    col_idx = pid % N
    
    # Accumulator for linear
    acc = 0.0
    
    # Compute linear: input @ weight.T
    # input shape: [batch, seq, K], weight shape: [K, N]
    # output shape: [batch, seq, N]
    # We need to iterate over K
    for k in range(0, K, BLOCK_SIZE):
        k_offsets = k + tl.arange(0, BLOCK_SIZE)
        mask_k = k_offsets < K
        
        # Load weight: weight[k, col_idx]
        w_offsets = k_offsets * N + col_idx
        w = tl.load(weight_ptr + w_offsets, mask=mask_k, other=0.0)
        
        # Load input: input[batch, seq, k]
        # For simplicity, treat input as [M, K] where M = batch * seq
        input_offsets = batch_idx * K + k_offsets
        x = tl.load(input_ptr + input_offsets, mask=mask_k, other=0.0)
        
        acc += tl.sum(x * w)
    
    # Store linear result
    out1_offset = batch_idx * N + col_idx
    tl.store(out1_ptr + out1_offset, acc)
    
    # Compute element-wise multiply: mul_input * scale (with broadcasting)
    # scale is [N] or scalar, mul_input is [M, N]
    scale_val = tl.load(scale_ptr + col_idx)
    mul_val = tl.load(mul_input_ptr + out1_offset)
    result = mul_val * scale_val
    
    # Store mul result
    tl.store(out2_ptr + out1_offset, result)


@torch.fx.wrap
def triton_linear_mul(weight, input_tensor, scale, mul_input):
    """
    Fused linear + element-wise multiply.
    
    Linear: output = input_tensor @ weight.T
    Mul: result = mul_input * scale (broadcast along last dim)
    
    Both operations are computed.
    Returns: (linear_output, mul_output)
    """
    # weight: [K, N], input_tensor: [B, S, K], output: [B, S, N]
    # scale: [N], mul_input: [B, S, N]
    
    B, S, K = input_tensor.shape
    N = weight.shape[0]
    M = B * S
    
    # Flatten for linear computation
    input_flat = input_tensor.view(M, K)
    mul_input_flat = mul_input.view(M, N)
    
    # Allocate outputs
    linear_out = torch.empty((B, S, N), dtype=input_tensor.dtype, device=input_tensor.device)
    mul_out = torch.empty((B, S, N), dtype=input_tensor.dtype, device=input_tensor.device)
    
    linear_out_flat = linear_out.view(M, N)
    mul_out_flat = mul_out.view(M, N)
    
    # Grid configuration
    BLOCK_SIZE = 128
    total_programs = M * N
    
    # Launch kernel
    triton_linear_mul_kernel[(total_programs,)](
        weight_ptr=weight,
        input_ptr=input_flat,
        scale_ptr=scale,
        mul_input_ptr=mul_input_flat,
        out1_ptr=linear_out_flat,
        out2_ptr=mul_out_flat,
        M=M, N=N, K=K,
        batch_stride=B * S * K if B > 1 else 0,  # not used directly
        seq_stride=S * K if S > 1 else 0,  # not used directly
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return linear_out, mul_out


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern: linear(in_3, in_0, None) followed by in_2 * in_1
    
    in_0: weight tensor [K, N]
    in_1: scale tensor [N]
    in_2: mul_input tensor [B, S, N]
    in_3: input tensor [B, S, K]
    
    Returns: (in_2 * in_1, linear_output)
    """
    linear = torch.nn.functional.linear(in_3, in_0, None)
    tmp_3 = in_2 * in_1
    return (tmp_3, linear)


def replacement_args(in_0, in_1, in_2, in_3):
    # For rtmpose pattern: in_0 is weight, in_1 is scale, in_2 is mul_input, in_3 is input
    # weight: [256, 512], scale: [256], mul_input: [B, 17, 256], input: [B, 17, 512]
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return triton_linear_mul