import torch
import triton
import triton.language as tl

# Pattern 3: Single-tensor identity (adding zeros) + mean reduction
def pattern(in_0):
    # Mirror the exact computation from model.py
    tmp_0 = 0 + in_0
    tmp_0 += 0
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def identity_mean_kernel(
    input_ptr,
    output_identity_ptr, output_mean_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Optimized grid-stride loop for identity + mean operation
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    total_elements = N * C * H * W
    
    for block_idx in range(pid, (total_elements + block_size - 1) // block_size, tl.num_programs(0)):
        start_idx = block_idx * block_size
        offsets = start_idx + tl.arange(0, block_size)
        mask = offsets < total_elements
        
        # Load input tensor (identity operation)
        input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        
        # Store identity result directly
        tl.store(output_identity_ptr + offsets, input_data, mask=mask)
        
        # Accumulate sum for mean calculation
        if tl.any(mask):
            partial_sum = tl.sum(input_data)
            # Use atomic add to safely accumulate partial sums
            tl.atomic_add(output_mean_ptr + 0, [partial_sum])

@torch.fx.wrap
def fused_identity_mean_single_tensor(input_tensor):
    N, C, H, W = input_tensor.shape
    
    # Create output tensors
    identity_out = torch.empty_like(input_tensor)
    mean_out = torch.empty((N, C, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Initialize mean accumulator
    mean_out.fill_(0.0)
    
    # Block size optimization based on spatial dimensions for better cache utilization
    if H * W >= 256:
        BLOCK_SIZE = 4096
    elif H * W >= 128:
        BLOCK_SIZE = 2048
    elif H * W >= 64:
        BLOCK_SIZE = 1024
    elif H * W >= 32:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 256
    
    # Calculate optimized grid size
    element_count = N * C * H * W
    grid_size = ((element_count + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch fused identity + mean kernel
    identity_mean_kernel[grid_size](
        input_tensor,
        identity_out, mean_out,
        N, C, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Finalize mean computation by division
    mean_out = mean_out / (H * W)
    
    return identity_out, mean_out

def replacement_func():
    return fused_identity_mean_single_tensor