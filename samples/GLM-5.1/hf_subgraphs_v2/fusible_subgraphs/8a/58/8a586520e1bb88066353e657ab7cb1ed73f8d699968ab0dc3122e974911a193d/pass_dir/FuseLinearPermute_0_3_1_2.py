import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.permute(0, 3, 1, 2)
    return tmp_3


def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)


@triton.jit
def fused_linear_permute_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Dimensions
    B, M1, M2, N,
    # Strides for input [B, M1, M2, K]
    input_stride_b, input_stride_m1, input_stride_m2, input_stride_k,
    # Strides for weight [N, K]
    weight_stride_n, weight_stride_k,
    # Strides for output [B, N, M1, M2] (permuted layout)
    output_stride_b, output_stride_n, output_stride_m1, output_stride_m2,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K: tl.constexpr,
):
    # 2D grid: each program handles a tile of output[b, n_block, m1_block, m2]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    
    # Decode combined spatial index to (m1, m2)
    m1 = offs_m // M2
    m2 = offs_m % M2
    
    mask_m = offs_m < M1 * M2
    mask_n = offs_n < N
    
    # Load bias tile [BLOCK_N]
    b_ptrs = bias_ptr + offs_n
    bias_vals = tl.load(b_ptrs, mask=mask_n, other=0.0).to(tl.float32)  # [BLOCK_N]
    
    # Accumulate dot product manually since K is small (3)
    # output[b, n, m1, m2] = sum_k(input[b, m1, m2, k] * weight[n, k]) + bias[n]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_idx in tl.static_range(K):
        # Load input[b, m1, m2, k_idx] -> [BLOCK_M]
        i_ptrs = input_ptr + pid_b * input_stride_b + m1 * input_stride_m1 + m2 * input_stride_m2 + k_idx * input_stride_k
        i_mask = mask_m
        inp_k = tl.load(i_ptrs, mask=i_mask, other=0.0).to(tl.float32)  # [BLOCK_M]
        
        # Load weight[n, k_idx] -> [BLOCK_N]
        w_ptrs = weight_ptr + offs_n * weight_stride_n + k_idx * weight_stride_k
        w_mask = mask_n
        w_k = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)  # [BLOCK_N]
        
        # acc[m, n] += inp_k[m] * w_k[n]
        acc += inp_k[:, None] * w_k[None, :]  # [BLOCK_M, BLOCK_N]
    
    acc += bias_vals[None, :]  # broadcast bias
    
    # Store output[b, n, m1, m2]
    o_ptrs = output_ptr + pid_b * output_stride_b + offs_n[None, :] * output_stride_n + m1[:, None] * output_stride_m1 + m2[:, None] * output_stride_m2
    o_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(o_ptrs, acc, mask=o_mask)


@torch.fx.wrap
def fused_linear_permute(bias, weight, input):
    # input shape: [B, M1, M2, K] = [1, 196, 196, 3]
    # weight shape: [N, K] = [16, 3]
    # bias shape: [N] = [16]
    # output shape: [B, N, M1, M2] = [1, 16, 196, 196] (permuted layout)
    
    B = input.size(0)
    M1 = input.size(1)
    M2 = input.size(2)
    K = input.size(3)
    N = weight.size(0)
    
    # Create output in permuted layout [B, N, M1, M2]
    output = torch.empty((B, N, M1, M2), dtype=input.dtype, device=input.device)
    
    BLOCK_M = 128  # tile along spatial dimension (M1*M2 combined)
    BLOCK_N = 16   # N=16, full coverage
    
    num_m_blocks = (M1 * M2 + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (N + BLOCK_N - 1) // BLOCK_N
    num_b_blocks = B
    
    grid = (num_m_blocks, num_n_blocks, num_b_blocks)
    
    fused_linear_permute_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        B=B, M1=M1, M2=M2, N=N,
        input_stride_b=input.stride(0), input_stride_m1=input.stride(1),
        input_stride_m2=input.stride(2), input_stride_k=input.stride(3),
        weight_stride_n=weight.stride(0), weight_stride_k=weight.stride(1),
        output_stride_b=output.stride(0), output_stride_n=output.stride(1),
        output_stride_m1=output.stride(2), output_stride_m2=output.stride(3),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        K=K,
    )
    
    return output


def replacement_func():
    return fused_linear_permute