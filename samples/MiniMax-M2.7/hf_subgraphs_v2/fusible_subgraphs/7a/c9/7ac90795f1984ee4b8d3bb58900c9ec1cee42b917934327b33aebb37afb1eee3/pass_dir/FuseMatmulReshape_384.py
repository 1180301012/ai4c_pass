import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 16}, num_stages=3, num_warps=1),
        triton.Config({'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=4),
    ],
    key=['K'],
)
@triton.jit
def matmul_reshape_kernel(
    in_1_ptr, in_0_ptr, out_ptr,
    B: tl.constexpr, M: tl.constexpr, K: tl.constexpr,
    stride_in_1_b, stride_in_1_m, stride_in_1_k,
    stride_in_0_b, stride_in_0_k,
    stride_out,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Fused matmul + reshape kernel for D=384."""
    batch_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    
    D = 384
    col_offsets = tl.arange(0, D)
    acc = tl.zeros((D,), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < K
        
        a_ptrs = (
            in_1_ptr + 
            batch_idx * stride_in_1_b + 
            row_idx * stride_in_1_m + 
            k_offsets * stride_in_1_k
        )
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)
        
        b_ptrs = (
            in_0_ptr + 
            batch_idx * stride_in_0_b +
            k_offsets * stride_in_0_k
        )
        b = tl.load(b_ptrs, mask=mask_k, other=0.0)
        
        acc += tl.sum(a * b)
    
    out_row_base = row_idx * D
    out_offsets = out_row_base + col_offsets
    out_ptrs = out_ptr + out_offsets
    tl.store(out_ptrs, acc)


@torch.fx.wrap
def fused_matmul_reshape(in_1, in_0):
    """Fused matmul + reshape for D=384."""
    B, M, K = in_1.shape
    D = 384
    total_elements = B * M
    num_rows = total_elements // D
    
    out = torch.empty((num_rows, D), dtype=in_1.dtype, device=in_1.device)
    grid = (B, num_rows)
    
    matmul_reshape_kernel[grid](
        in_1, in_0, out,
        B, M, K,
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        in_0.stride(0), in_0.stride(1),
        out.stride(0),
    )
    return out


def pattern(in_1, in_0, in_2):
    """Pattern for matmul + reshape to [-1, 384]."""
    matmul = torch.matmul(in_1, in_0)
    tmp_1 = torch.reshape(matmul, [-1, 384])
    return tmp_1


def replacement_args(in_1, in_0, in_2):
    return (in_1, in_0, "D384")


def replacement_func():
    def router(in_1, in_0, route):
        assert route == "D384", f"Expected route D384, got {route}"
        return fused_matmul_reshape(in_1, in_0)
    return router