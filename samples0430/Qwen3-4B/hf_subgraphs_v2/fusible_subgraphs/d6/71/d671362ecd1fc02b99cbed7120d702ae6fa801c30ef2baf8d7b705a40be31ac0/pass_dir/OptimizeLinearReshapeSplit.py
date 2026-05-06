import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_2, in_1)
    reshaped = linear.reshape(1, 49, 8, -1)
    split0, split1, split2 = reshaped.split([32, 32, 128], dim=3)
    permuted0 = split0.permute(0, 2, 1, 3)
    permuted1 = split1.permute(0, 2, 1, 3)
    permuted2 = split2.permute(0, 2, 1, 3)
    gpu_in0 = in_0.to(device(type='cuda', index=0))
    transposed = permuted1.transpose(-2, -1)
    return (permuted0, gpu_in0, transposed, permuted2)

def replacement_args(in_0, in_1, in_2, in_3):
    return in_0, in_1, in_2, in_3

@triton.jit
def optimized_kernel(
    in_3_ptr,
    in_2_ptr,
    in_1_ptr,
    out_0_ptr,
    out_1_ptr,
    out_2_ptr,
    out_3_ptr,
    BLOCK_SIZE: tl.constexpr
):
    block_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    in_3 = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
    in_2 = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
    in_1 = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
    out_0 = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
    out_1 = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
    out_2 = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
    out_3 = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
    tl.store(out_0_ptr + offsets, out_0, mask=offsets < BLOCK_SIZE)
    tl.store(out_1_ptr + offsets, out_1, mask=offsets < BLOCK_SIZE)
    tl.store(out_2_ptr + offsets, out_2, mask=offsets < BLOCK_SIZE)
    tl.store(out_3_ptr + offsets, out_3, mask=offsets < BLOCK_SIZE)

torch.fx.wrap
@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, in_3):
    batch_size = 1
    seq_len = 49
    num_heads = 8
    out_dim = 192
    
    out_0 = torch.empty((batch_size, seq_len, num_heads, out_dim), 
                       dtype=torch.bfloat16, 
                       device='cuda')
    out_1 = in_0.to(device='cuda')
    out_2 = torch.empty((batch_size, seq_len, out_dim), 
                       dtype=torch.bfloat16, 
                       device='cuda')
    out_3 = torch.empty((batch_size, seq_len, num_heads, out_dim), 
                       dtype=torch.bfloat16, 
                       device='cuda')
    
    grid = (1, 1, 1)
    optimized_kernel[grid](
        in_3_ptr=in_3,
        in_2_ptr=in_2,
        in_1_ptr=in_1,
        out_0_ptr=out_0,
        out_1_ptr=out_1,
        out_2_ptr=out_2,
        out_3_ptr=out_3,
        BLOCK_SIZE=128
    )
    return (out_0, out_1, out_2, out_3)

def replacement_func():
    return kernel_wrapper