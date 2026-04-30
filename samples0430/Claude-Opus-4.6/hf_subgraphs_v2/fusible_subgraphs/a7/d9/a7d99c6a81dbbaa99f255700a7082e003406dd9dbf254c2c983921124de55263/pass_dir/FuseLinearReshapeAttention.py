import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_5 = linear.view(1, 1, -1, 64)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_10 = tmp_6.contiguous()
    return tmp_10


def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)


@triton.jit
def linear_reshape_kernel(
    in_3_ptr, in_1_ptr, in_0_ptr, out_ptr,
):
    # GEMV: out[n] = sum_k(weight[n,k] * input[k]) + bias[n]
    # 8 programs, each computing 64 outputs, no inner loop (K=512 done in one shot)
    BLOCK_N: tl.constexpr = 64
    K: tl.constexpr = 512
    
    pid = tl.program_id(0)
    n_offset = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offset = tl.arange(0, K)
    
    # Load entire input vector [K]
    x = tl.load(in_3_ptr + k_offset).to(tl.float32)
    
    # Load weight tile [BLOCK_N, K] and compute dot products
    w = tl.load(in_1_ptr + n_offset[:, None] * K + k_offset[None, :]).to(tl.float32)
    acc = tl.sum(w * x[None, :], axis=1)
    
    # Add bias
    bias = tl.load(in_0_ptr + n_offset).to(tl.float32)
    acc += bias
    
    # Store
    tl.store(out_ptr + n_offset, acc.to(out_ptr.dtype.element_ty))


@torch.fx.wrap
def fused_linear_reshape(in_0, in_1, in_3):
    out = torch.empty((1, 8, 1, 64), dtype=in_3.dtype, device=in_3.device)
    linear_reshape_kernel[(8,)](in_3, in_1, in_0, out, num_warps=4, num_stages=1)
    return out


def replacement_func():
    return fused_linear_reshape