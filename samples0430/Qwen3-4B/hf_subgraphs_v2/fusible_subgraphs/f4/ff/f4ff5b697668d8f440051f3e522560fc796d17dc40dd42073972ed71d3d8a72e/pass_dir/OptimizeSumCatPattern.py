import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    softmax = torch.nn.functional.softmax(in_2, dim=2)
    tmp_3 = softmax.reshape(-1, 17, 64, 64)
    tmp_4 = tmp_3 * in_0
    tmp_5 = tmp_4.reshape(-1, 17, -1)
    tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
    tmp_7 = tmp_3 * in_1
    tmp_8 = tmp_7.reshape(-1, 17, -1)
    tmp_9 = torch.sum(tmp_8, dim=2, keepdim=True)
    tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
    return tmp_3, tmp_10

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def optimized_kernel(
    in0_ptr, in1_ptr, in2_ptr,
    out3_ptr, out10_ptr,
    B: tl.int32, H: tl.int32, W: tl.int32,
    num_spatial: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    # Compute block index
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, B)
    
    # Process each batch element
    for i in range(block_start, block_end):
        # Load tensors
        in2 = tl.load(in2_ptr + i * (H * W) * num_spatial, tl.type(float))
        
        # Compute for in_0 and in_1
        in0_val = tl.load(in0_ptr + i)
        in1_val = tl.load(in1_ptr + i)
        
        # Process and store results
        out3 = in2.reshape(17, W, W)
        tl.store(out3_ptr + i * (17 * W * W), out3)
        
        # Compute and store the two sums
        sum0 = tl.sum(out3, dim=2, keepdim=True)
        sum1 = tl.sum(out3, dim=2, keepdim=True)
        tl.store(out10_ptr + i * 17 * 2, tl.cat([sum0, sum1], dim=-1))

@torch.fx.wrap
def kernel_wrapper(in0, in1, in2):
    B = in2.shape[0]
    H = 17
    W = 64
    num_spatial = in2.shape[2] // (W * W)
    
    out3 = torch.empty((B, H, W, W), dtype=in2.dtype, device=in2.device)
    out10 = torch.empty((B, H, 2), dtype=in2.dtype, device=in2.device)
    
    # Launch kernel with appropriate grid
    optimized_kernel[(tl.cdiv(B, BLOCK_SIZE), 1)](  
        in0_ptr=in0,
        in1_ptr=in1,
        in2_ptr=in2,
        out3_ptr=out3,
        out10_ptr=out10,
        B=B, H=H, W=W, num_spatial=num_spatial,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out3, out10

def replacement_func():
    return kernel_wrapper