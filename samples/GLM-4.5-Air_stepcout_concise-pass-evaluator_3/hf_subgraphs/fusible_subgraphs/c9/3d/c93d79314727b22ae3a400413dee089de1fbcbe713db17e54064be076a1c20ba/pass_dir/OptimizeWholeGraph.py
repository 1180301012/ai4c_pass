import torch
import triton
import triton.language as tl

def pattern(tmp_4, tmp_3, tmp_2, in_0, in_1):
    # Match the complete computation structure:
    # tmp_0 = in_0
    # tmp_1 = in_1  
    # tmp_2 = torch.matmul(in_2, in_3)
    # tmp_3 = tmp_1.to(device(type='cuda'))
    # tmp_4 = tmp_0.to(device(type='cuda'))
    # return (tmp_4, tmp_3, tmp_2)
    
    # Since we want to optimize the entire forward pass, we need to match
    # the observable inputs and outputs. However, for this specific case,
    # we'll focus on optimizing the matmul and return all required outputs
    
    # We can't match the entire graph at once because of the device transfers,
    # so this pass will be designed to catch the overall computation pattern
    # that includes the matmul and the observable return values
    
    # Note: This pass might match the same computation as the first pass
    # but with a different pattern scope
    return (tmp_4, tmp_3, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

# This is essentially the same as the first pass since the main optimization
# benefit comes from the matmul operation
@triton.jit
def matmul_vector_kernel(
    a_ptr,       # [M, K]
    b_ptr,       # [K] (flattened)
    out_ptr,     # [M] (flattened)
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for parallel execution
    pid = tl.program_id(0)
    m = pid
    
    # Check bounds
    if m >= M:
        return
    
    # Use power-of-2 block size (1024) for efficient GPU computation
    acc = 0.0
    for k in range(0, K, BLOCK_SIZE):
        # Calculate current offset with bounds masking
        offsets = k + tl.arange(0, BLOCK_SIZE)
        mask = offsets < K
        
        # Load vectors for this iteration
        a_vec = tl.load(a_ptr + m * K + offsets, mask=mask, other=0.0)
        b_vec = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        
        # Accumulate dot product
        acc += tl.sum(a_vec * b_vec)
    
    # Store the result
    tl.store(out_ptr + m, acc)

@torch.fx.wrap
def optimized_matmul_complete(a, b):
    # This function is identical to the one in the first pass
    # since the optimization target is the same
    M, K = a.shape
    
    # Use power-of-2 block size (1024) for GPU efficiency
    BLOCK_SIZE = 1024
    
    # Create output tensor - we'll create it as [M, 1] and flatten for kernel
    out_flat = torch.empty((M,), dtype=a.dtype, device=a.device)
    
    num_programs = M
    matmul_vector_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b.flatten(),
        out_ptr=out_flat,   # Store flattened result
        M=M,
        K=K,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to [M, 1]
    return out_flat.reshape(M, 1)

def replacement_func():
    return optimized_matmul_complete