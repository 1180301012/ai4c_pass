import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_1)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def optimized_kernel(input_ptr, output_ptr, mean_ptr, N, C, H, W, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < H * W
    
    input_vals = tl.load(input_ptr + (pid * BLOCK_SIZE) + offsets, mask=mask, other=0.0)
    gelu_vals = tl.zeros_like(input_vals)
    tl.store(output_ptr + (pid * BLOCK_SIZE) + offsets, gelu_vals)
    
    sum_vals = tl.sum(gelu_vals, mask=mask)
    mean_vals = sum_vals / (H * W)
    tl.store(mean_ptr + pid, mean_vals)

def kernel_wrapper(in_0):
    N = in_0.shape[0]
    C = in_0.shape[1]
    H = in_0.shape[2]
    W = in_0.shape[3]
    
    output = torch.empty_like(in_0)
    mean_tensor = torch.zeros((N, C), dtype=in_0.dtype)
    
    BLOCK_SIZE = 256
    num_blocks = (N * C * H * W + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_kernel[(num_blocks,)](  
        input_ptr=in_0,
        output_ptr=output,
        mean_ptr=mean_tensor,
        N=N,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    mean_tensor = mean_tensor / (H * W)
    return (output, mean_tensor.unsqueeze(-1).unsqueeze(-1))

def replacement_func():
    return kernel_wrapper