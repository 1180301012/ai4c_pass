import torch
import triton
import triton.language as tl

@triton.jit
def chunk_kernel(
    input_ptr,
    output0_ptr,
    output1_ptr,
    N, C_total, H, W,
    chunk_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element
    pid = tl.program_id(0)
    
    if pid >= N * C_total * H * W:
        return
        
    # Calculate indices
    batch_idx = pid // (C_total * H * W)
    c_total = (pid // (H * W)) % C_total
    h = (pid // W) % H
    w = pid % W
    
    # Determine which chunk this belongs to and the local index
    chunk_idx = c_total // chunk_size
    local_c = c_total % chunk_size
    
    if batch_idx < N and c_total < C_total and h < H and w < W:
        input_idx = pid
        
        # Load input value
        val = tl.load(input_ptr + input_idx)
        
        # Store to appropriate chunk
        if chunk_idx == 0:
            # First chunk: output0[batch_idx, local_c, h, w]
            output0_idx = batch_idx * chunk_size * H * W + local_c * H * W + h * W + w
            tl.store(output0_ptr + output0_idx, val)
        elif chunk_idx == 1:
            # Second chunk: output1[batch_idx, local_c, h, w]
            output1_idx = batch_idx * chunk_size * H * W + local_c * H * W + h * W + w
            tl.store(output1_ptr + output1_idx, val)

@torch.fx.wrap  
def optimized_chunk(input_tensor):
    # Original: chunk(2, dim=1) on tensor of shape [N, C_total, H, W]
    # Returns tuple of [N, C_total//2, H, W] tensors
    
    N, C_total, H, W = input_tensor.shape
    chunk_size = C_total // 2  # We're always chunking into 2 along dim=1
    
    # Create output tensors
    output0 = torch.empty((N, chunk_size, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    output1 = torch.empty((N, chunk_size, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Check if we need to do any work
    if N == 0 or H == 0 or W == 0:
        return output0, output1
    
    BLOCK_SIZE = 256
    grid_size = N * C_total * H * W
    num_programs = (grid_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    chunk_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output0_ptr=output0,
        output1_ptr=output1,
        N=N, C_total=C_total, H=H, W=W,
        chunk_size=chunk_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output0, output1

def pattern(x):
    # Original pattern: chunk(2, dim=1) and then access [0] and [1]
    chunk_result = x.chunk(2, dim=1)
    part0 = chunk_result[0]
    part1 = chunk_result[1]  
    return part0, part1

def replacement_args(x):
    return (x,)

def replacement_func():
    return optimized_chunk