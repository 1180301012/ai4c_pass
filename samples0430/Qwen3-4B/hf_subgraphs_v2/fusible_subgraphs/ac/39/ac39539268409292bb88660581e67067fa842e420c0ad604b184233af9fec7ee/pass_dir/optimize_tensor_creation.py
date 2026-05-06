import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    N = 32
    meshgrid_i = torch.arange(N)
    meshgrid_j = torch.arange(N)
    meshgrid = torch.meshgrid(meshgrid_i, meshgrid_j, indexing='ij')
    stack = torch.stack((meshgrid[0], meshgrid[1]))
    flattened = torch.flatten(stack, 1).unsqueeze(2)
    
    tmp8 = flattened[:, :, 0]
    tmp9 = flattened[:, 0, :]
    tmp10 = tmp8 - tmp9
    
    permuted = tmp10.permute(1, 2, 0)
    permuted[:, :, 0] += (N - 1)
    permuted[:, :, 1] += (N - 1)
    permuted[:, :, 0] *= 63
    
    output_size = (N + 1, N + 1)
    output = torch.zeros(output_size, dtype=torch.int64)
    output[1:, 1:] = permuted.sum(-1)
    output[0, 1:] = 3969
    output[1:, 0] = 3970
    output[0, 0] = 3971
    return torch.cat([in_1, in_0]), output.view(-1)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    return kernel_wrapper

@triton.jit
def optimized_tensor_kernel(
    output_ptr: tl.int64,
    N: tl.constexpr,
    block_id: tl.int32,
    block_size: tl.constexpr,
    output_dtype: tl.int32
):
    block_start = block_id * block_size
    offsets = block_start + tl.arange(0, block_size, dtype=tl.int32)
    mask = offsets < (N + 1) * (N + 1)
    
    output = tl.zeros(block_size, dtype=output_dtype)
    
    for i in tl.arange(0, block_size):
        idx = block_start + i
        r = idx // (N + 1)
        c = idx % (N + 1)
        
        if r == 0 and c == 0:
            output[i] = 3971
        elif r == 0:
            output[i] = 3969
        elif c == 0:
            output[i] = 3970
        else:
            output[i] = (r + c) * 63
    
    tl.store(output_ptr + offsets, output, mask=mask)

def kernel_wrapper(N, in_0, in_1):
    output_size = (N + 1) * (N + 1)
    output = torch.empty(output_size, dtype=torch.int64)
    
    num_blocks = (output_size + 1024 - 1) // 1024
    optimized_tensor_kernel[num_blocks](
        output_ptr=output,
        N=N,
        block_id=tl.int32,
        block_size=1024,
        output_dtype=torch.int64
    )
    return output.view(-1)