import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_2 = in_0 * in_2
    tmp_4 = tmp_2.float()
    tmp_5 = tmp_4.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_9 = tmp_4 * tmp_8
    tmp_10 = in_1.float()
    tmp_11 = 1.0 + tmp_10
    tmp_12 = tmp_9 * tmp_11
    tmp_13 = tmp_12.type_as(tmp_2)
    return (tmp_2, tmp_13)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def optimized_kernel(
    in_0_ptr: tl.float32,
    in_2_ptr: tl.float32,
    in_1_ptr: tl.float32,
    out_0_ptr: tl.float32,
    out_1_ptr: tl.float32,
    N: tl.int32,
    M: tl.int32,
    BLOCK_SIZE: tl.int32,
):
    # Grid implementation for 1x3x2048 tensor
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE
    mask = tl.arange(0, BLOCK_SIZE) < N
    
    # Initialize accumulators
    sum_sq = tl.zeros(tl.float32)
    
    # Process each feature dimension
    for i in tl.arange(0, N):
        
        # Compute position
        pos = offs + i
        
        # Get input values
        x = tl.load(in_0_ptr + pos, mask=mask, other=0.0)
        x_scaled = x * in_2_ptr
        x_sq = x_scaled * x_scaled
        
        # Accumulate sum of squares
        sum_sq += x_sq
        
    # Compute normalization
    mean_sq = sum_sq / N
    norm = 1.0 / tl.sqrt(mean_sq + 1e-06)
    
    # Process weights
    w = tl.load(in_1_ptr + i, mask=mask, other=0.0)
    w = w + 1.0
    
    # Store results
    tl.store(out_0_ptr + pos, x_scaled)
    tl.store(out_1_ptr + pos, x_scaled * norm * w)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2):
    N = in_0.shape[2]
    M = in_0.shape[1]
    BLOCK_SIZE = 128
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out_0 = torch.empty_like(in_0)
    out_1 = torch.empty_like(in_0)
    
    optimized_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_2_ptr=in_2,
        in_1_ptr=in_1,
        out_0_ptr=out_0,
        out_1_ptr=out_1,
        N=N,
        M=M,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out_0, out_1)

def replacement_func():
    return kernel_wrapper