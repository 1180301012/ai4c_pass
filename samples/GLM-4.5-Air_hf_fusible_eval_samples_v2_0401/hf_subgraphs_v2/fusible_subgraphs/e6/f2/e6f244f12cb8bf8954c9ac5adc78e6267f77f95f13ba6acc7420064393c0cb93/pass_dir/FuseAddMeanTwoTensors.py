import torch
import triton
import triton.language as tl

# Pattern 2: Two-tensor addition (with zero initialization) + mean reduction
def pattern(in_0, in_1):
    # Mirror the exact computation from model.py  
    tmp_0 = 0 + in_1
    tmp_0 += in_0
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_add_mean_kernel_2(
    in0_ptr, in1_ptr,
    out_sum_ptr, out_mean_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid-stride loop with optimized blocks
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    total_elements = N * C * H * W
    
    for block_idx in range(pid, (total_elements + block_size - 1) // block_size, tl.num_programs(0)):
        start_idx = block_idx * block_size
        offsets = start_idx + tl.arange(0, block_size)
        mask = offsets < total_elements
        
        # Load input tensors
        in0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
        in1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)
        
        # Fused addition
        sum_result = in0 + in1
        
        # Store sum directly
        tl.store(out_sum_ptr + offsets, sum_result, mask=mask)
        
        # Accumulate sum for mean calculation (atomic operation)
        if tl.any(mask):
            partial_sum = tl.sum(sum_result)
            # Use atomic add to accumulate mean
            tl.atomic_add(out_mean_ptr + 0, [partial_sum])

@torch.fx.wrap
def fused_add_mean_2_tensors(in0, in1):
    N, C, H, W = in0.shape
    
    # Create output tensors
    sum_out = torch.empty_like(in0)
    mean_out = torch.empty((N, C, 1, 1), dtype=in0.dtype, device=in0.device)
    
    # Initialize mean accumulator
    mean_out.fill_(0.0)
    
    # Block size optimization based on tensor dimensions
    if H * W >= 128:
        BLOCK_SIZE = 2048
    elif H * W >= 64:
        BLOCK_SIZE = 1024
    elif H * W >= 32:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = (triton.cdiv(N * C * H * W, BLOCK_SIZE),)
    
    # Launch fused kernel
    fused_add_mean_kernel_2[grid_size](
        in0, in1,
        sum_out, mean_out,
        N, C, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Finalize mean computation
    mean_out = mean_out / (H * W)
    
    return sum_out, mean_out

def replacement_func():
    return fused_add_mean_2_tensors