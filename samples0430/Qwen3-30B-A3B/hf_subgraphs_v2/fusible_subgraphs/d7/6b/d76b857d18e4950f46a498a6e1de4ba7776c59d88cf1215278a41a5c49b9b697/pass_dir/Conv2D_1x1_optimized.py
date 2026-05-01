import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor):
    return torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

@triton.jit
def conv2d_1x1_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr,
                     M, N, K,
                     BLOCK_SIZE_N: tl.constexpr):
    m = tl.program_id(0)
    row_start = m * K
    n = tl.thread_id(0)
    if n >= N:
        return
    acc = 0.0
    for k in range(0, K):
        input_val = tl.load(input_ptr + row_start + k)
        weight_val = tl.load(weight_ptr + n * K + k)
        acc += input_val * weight_val
    acc += tl.load(bias_ptr + n)
    output_idx = m * N + n
    tl.store(output_ptr + output_idx, acc)

@torch.fx.wrap
def conv2d_1x1_optimized(input_tensor, weight_tensor, bias_tensor):
    batch_size = input_tensor.shape[0]
    H = input_tensor.shape[2]
    W = input_tensor.shape[3]
    M = batch_size * H * W
    K = input_tensor.shape[1]
    N = weight_tensor.shape[0]
    
    input_reshaped = input_tensor.permute(0, 2, 3, 1).reshape(M, K)
    output_reshaped = torch.empty((M, N), dtype=input_tensor.dtype, device=input_tensor.device)
    
    BLOCK_SIZE_N = 32
    grid = (M, )
    conv2d_1x1_kernel[grid](
        input_ptr=input_reshaped,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        output_ptr=output_reshaped,
        M=M,
        N=N,
        K=K,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    output = output_reshaped.view(batch_size, H, W, N).permute(0, 3, 1, 2)
    return output

def replacement_func():
    return conv2d_1x1_optimized