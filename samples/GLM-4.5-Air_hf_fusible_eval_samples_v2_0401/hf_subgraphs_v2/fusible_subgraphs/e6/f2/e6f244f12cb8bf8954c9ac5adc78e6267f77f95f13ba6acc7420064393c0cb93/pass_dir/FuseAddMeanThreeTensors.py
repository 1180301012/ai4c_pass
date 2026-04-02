import torch
import triton
import triton.language as tl

# Pattern 1: Three-tensor addition + mean reduction
def pattern(in_0, in_1, in_2):
    # Mirror the exact computation from model.py
    tmp_0 = in_1 + in_2
    tmp_0 += in_0
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_add_mean_kernel_3(
    in0_ptr, in1_ptr, in2_ptr,
    out_sum_ptr, out_mean_ptr,
    N, H, W,
    BLOCK_SIZE: tl.constexpr,
    USE_FUSED_KERNEL: tl.constexpr,
):
    # Grid-stride loop
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    num_elements = N * H * W
    total_blocks = (num_elements + block_size - 1) // block_size
    
    for block_idx in range(pid, total_blocks, tl.num_programs(0)):
        start_idx = block_idx * block_size
        end_idx = min(start_idx + block_size, num_elements)
        offsets = start_idx + tl.arange(0, end_idx - start_idx)
        mask = offsets < num_elements
        
        # Load all three input tensors
        in0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
        in1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)
        in2 = tl.load(in2_ptr + offsets, mask=mask, other=0.0)
        
        # Fused addition
        sum_result = in0 + in1 + in2
        
        # Store both sum and mean in one pass if fused
        tl.store(out_sum_ptr + offsets, sum_result, mask=mask)
        
        # For mean, we'll handle separately in host code to avoid complex reduction in kernel
        if offsets[0] == 0:
            # Use atomic add for mean accumulation
            tl.store(out_mean_ptr + offsets[:1], [tl.sum(sum_result)], mask=offsets[:1] < 1)

@torch.fx.wrap
def fused_add_mean_3_tensors(in0, in1, in2):
    N, C, H, W = in0.shape
    
    # Create output tensors
    sum_out = torch.empty_like(in0)
    mean_out = torch.empty((N, C, 1, 1), dtype=in0.dtype, device=in0.device)
    
    # Initialize mean accumulator
    mean_out.fill_(0.0)
    
    # Block size optimization for different tensor sizes
    if H * W >= 64:
        BLOCK_SIZE = 1024
    elif H * W >= 32:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 256
    
    grid = lambda meta: (triton.cdiv(N * C * H * W, BLOCK_SIZE),)
    
    # Launch fused kernel
    fused_add_mean_kernel_3[grid](
        in0, in1, in2,
        sum_out, mean_out,
        N, H, W,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_FUSED_KERNEL=True
    )
    
    # Finalize mean computation by dividing by H*W
    mean_out = mean_out / (H * W)
    
    return sum_out, mean_out

def replacement_func():
    return fused_add_mean_3_tensors