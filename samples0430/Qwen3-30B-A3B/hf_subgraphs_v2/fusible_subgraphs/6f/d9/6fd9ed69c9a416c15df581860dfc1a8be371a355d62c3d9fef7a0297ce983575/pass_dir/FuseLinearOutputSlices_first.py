import torch
import triton
import triton.language as tl

def pattern(in_5, in_1, in_0):
    linear = torch.nn.functional.linear(in_5, in_1, in_0)
    tmp_5 = linear[(slice(None, None, None), slice(None, 256, None))]
    tmp_7 = linear[(slice(None, None, None), slice(-256, None, None))]
    return tmp_5, tmp_7

def replacement_args(in_5, in_1, in_0):
    return (in_5, in_1, in_0)

@triton.jit
def fused_linear_kernel(
    in_ptr,
    weight1_ptr,
    weight2_ptr,
    bias1_ptr,
    bias2_ptr,
    out1_ptr,
    out2_ptr,
    n,
    m,
    k,
    BLOCK_M: tl.constexpr=256,
    BLOCK_K: tl.constexpr=64
):
    i = tl.program_id(0)
    acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    bias1 = tl.load(bias1_ptr + tl.arange(0, BLOCK_M), mask=tl.arange(0, BLOCK_M) < m, other=0.0)
    bias2 = tl.load(bias2_ptr + tl.arange(0, BLOCK_M), mask=tl.arange(0, BLOCK_M) < m, other=0.0)
    
    for k_start in range(0, k, BLOCK_K):
        in_tile = tl.load(
            in_ptr + i * k + k_start,
            mask=tl.arange(0, BLOCK_K) < k - k_start,
            other=0.0
        )
        weight1_tile = tl.load(
            weight1_ptr + k_start * m,
            mask=(tl.arange(0, BLOCK_K)[:, None] < k - k_start) & (tl.arange(0, m) < m),
            other=0.0
        )
        weight2_tile = tl.load(
            weight2_ptr + k_start * m,
            mask=(tl.arange(0, BLOCK_K)[:, None] < k - k_start) & (tl.arange(0, m) < m),
            other=0.0
        )
        
        acc1 += tl.dot(in_tile, weight1_tile)
        acc2 += tl.dot(in_tile, weight2_tile)
    
    out1 = acc1 + bias1
    out2 = acc2 + bias2
    tl.store(out1_ptr + i * m, out1, mask=tl.arange(0, BLOCK_M) < m)
    tl.store(out2_ptr + i * m, out2, mask=tl.arange(0, BLOCK_M) < m)

@torch.fx.wrap
def fused_linear_kernel_wrapper(in_5, in_1, in_0):
    weight1 = in_1[:256, :]
    weight2 = in_1[256:, :]
    bias1 = in_0[:256]
    bias2 = in_0[256:]
    
    n = in_5.shape[0]
    m = 256
    k = in_5.shape[1]
    
    out1 = torch.empty((n, m), dtype=in_5.dtype, device=in_5.device)
    out2 = torch.empty((n, m), dtype=in_5.dtype, device=in_5.device)
    
    fused_linear_kernel[(n,)](
        in_5,
        weight1,
        weight2,
        bias1,
        bias2,
        out1,
        out2,
        n,
        m,
        k,
        BLOCK_M=256,
        BLOCK_K=64
    )
    return out1, out2

def replacement_func():
    return fused_linear_kernel_wrapper