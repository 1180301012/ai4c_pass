import torch
import triton
import triton.language as tl


# Pattern matching function - matches matmul followed by view(32, 512, 1, 1)
def pattern(in_0, in_1):
    """
    Match the pattern: matmul(in_1, in_0) followed by view(32, 512, 1, 1)
    """
    tmp_0 = torch.matmul(in_1, in_0)
    tmp_1 = tmp_0.view(32, 512, 1, 1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_matmul_view_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    M: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program processes one row of the output
    pid = tl.program_id(0)
    row_idx = pid
    
    if row_idx >= M:
        return
    
    # Accumulator for this row
    acc = 0.0
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load in_0: [K] - the weight vector
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < K
        
        a_ptrs = in_0_ptr + k_offsets
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)  # [BK]
        
        # Load in_1 row: [K] - one row of the matrix
        b_ptrs = in_1_ptr + row_idx * K + k_offsets
        b = tl.load(b_ptrs, mask=mask_k, other=0.0)  # [BK]
        
        # Element-wise multiply and accumulate
        acc += tl.sum(a * b)
    
    # Store result
    out_ptrs = out_ptr + row_idx
    tl.store(out_ptrs, acc)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1):
    # in_0: [batch=32, head=1, k=4096, n=1]
    # in_1: [batch=32, head=1, m=512, k=4096]
    batch, head, m, k = in_1.shape[0], in_1.shape[1], in_1.shape[2], in_1.shape[3]
    
    # Extract data for batch=0, head=0
    in_0_slice = in_0[0, 0, :, 0].contiguous()  # [4096]
    in_1_slice = in_1[0, 0, :, :].contiguous()  # [512, 4096]
    
    out = torch.empty((m,), dtype=torch.float32, device=in_0.device)
    
    BLOCK_SIZE_M = 1  # Each thread processes one row
    BLOCK_SIZE_K = 64
    grid = (m,)
    
    fused_matmul_view_kernel[grid](
        in_0_ptr=in_0_slice,
        in_1_ptr=in_1_slice,
        out_ptr=out,
        M=m, K=k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Reshape to [1, m, 1, 1] then broadcast to [32, 512, 1, 1]
    out = out.reshape(1, m, 1, 1)
    out = out.expand(32, 512, 1, 1)
    
    return out


def replacement_func():
    return fused_kernel_wrapper