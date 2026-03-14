import torch
import triton
import triton.language as tl


# Pattern matching function - matches matmul followed by view(1, 80, 1, 1)
def pattern(in_0, in_1):
    """
    Match the pattern: matmul(in_1, in_0) followed by view(1, 80, 1, 1)
    """
    tmp_0 = torch.matmul(in_1, in_0)
    tmp_1 = tmp_0.view(1, 80, 1, 1)
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
    pid = tl.program_id(0)
    row_idx = pid
    
    if row_idx >= M:
        return
    
    acc = 0.0
    
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < K
        
        a_ptrs = in_0_ptr + k_offsets
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)
        
        b_ptrs = in_1_ptr + row_idx * K + k_offsets
        b = tl.load(b_ptrs, mask=mask_k, other=0.0)
        
        acc += tl.sum(a * b)
    
    out_ptrs = out_ptr + row_idx
    tl.store(out_ptrs, acc)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1):
    batch, head, m, k = in_1.shape[0], in_1.shape[1], in_1.shape[2], in_1.shape[3]
    
    in_0_slice = in_0[0, 0, :, 0].contiguous()
    in_1_slice = in_1[0, 0, :, :].contiguous()
    
    out = torch.empty((m,), dtype=torch.float32, device=in_0.device)
    
    BLOCK_SIZE_M = 1
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
    
    out = out.reshape(1, m, 1, 1)
    return out


def replacement_func():
    return fused_kernel_wrapper