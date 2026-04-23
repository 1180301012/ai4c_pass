import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    return tmp_3

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def fused_linear_sigmoid_kernel(
    in_2_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    B,
    CH,
    IN,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m_start = tl.program_id(0) * BLOCK_M
    n_start = tl.program_id(1) * BLOCK_N
    
    m = m_start + tl.arange(0, BLOCK_M)
    n = n_start + tl.arange(0, BLOCK_N)
    
    mask_m = m < CH
    mask_n = n < B
    mask = mask_m[:, None] & mask_n[None, :]
    
    in_2 = tl.load(
        in_2_ptr + (n[:, None] * IN + tl.arange(0, IN)),
        mask=mask_n[:, None],
        other=0.0
    )
    
    weight = tl.load(
        weight_ptr + (m[None, :] * IN + tl.arange(0, IN)),
        mask=mask_m[None, :],
        other=0.0
    )
    
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k in range(IN):
        in_2_k = in_2[:, k]
        weight_k = weight[:, k]
        acc += in_2_k[:, None] * weight_k[None, :]
    
    acc = tl.sum(acc, axis=1)
    bias = tl.load(bias_ptr + m, mask=mask_m, other=0.0)
    acc += bias
    acc = 1.0 / (1.0 + tl.exp(-acc))
    
    tl.store(
        out_ptr + (n[:, None] * CH + m),
        acc,
        mask=mask
    )

@torch.fx.wrap
def kernel_wrapper(in_2, in_1, in_0):
    B = in_2.shape[0]
    CH = in_1.shape[0]
    IN = in_1.shape[1]
    
    out = torch.empty((B, CH), dtype=in_2.dtype)
    
    BLOCK_M = 32
    BLOCK_N = 16
    
    grid_m = (CH + BLOCK_M - 1) // BLOCK_M
    grid_n = (B + BLOCK_N - 1) // BLOCK_N
    
    fused_linear_sigmoid_kernel[(grid_m, grid_n)](
        in_2,
        in_1,
        in_0,
        out,
        B,
        CH,
        IN,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )
    
    return out

def replacement_func():
    return kernel_wrapper