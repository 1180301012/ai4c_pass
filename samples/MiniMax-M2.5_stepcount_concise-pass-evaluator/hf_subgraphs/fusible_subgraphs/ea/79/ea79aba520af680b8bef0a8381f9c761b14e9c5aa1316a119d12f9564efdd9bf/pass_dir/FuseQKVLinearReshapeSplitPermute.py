import torch
from torch import device
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """Match QKV linear + reshape + split pattern."""
    tmp_3 = torch.nn.functional.linear(in_3, in_2, in_1)
    tmp_4 = tmp_3.reshape(64, 49, 8, -1)
    tmp_5 = tmp_4.split([32, 32, 128], dim=3)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    tmp_9 = tmp_6.permute(0, 2, 1, 3)
    tmp_10 = tmp_7.permute(0, 2, 1, 3)
    tmp_11 = tmp_8.permute(0, 2, 1, 3)
    tmp_12 = in_0.to(device(type='cuda', index=0))
    tmp_13 = tmp_10.transpose(-2, -1)
    return (tmp_9, tmp_12, tmp_13, tmp_11)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def triton_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K, stride_im, stride_ik, stride_wk, stride_wn, stride_om, stride_on, BLOCK_SIZE: tl.constexpr
):
    """Simple Triton linear kernel."""
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    if row >= M or col >= N:
        return
    
    acc = 0.0
    for k in range(0, K, BLOCK_SIZE):
        k_remain = min(BLOCK_SIZE, K - k)
        
        offs_k = tl.arange(0, BLOCK_SIZE)
        mask_k = offs_k < k_remain
        
        offs_im = row * stride_im + (k + offs_k) * stride_ik
        offs_wk = col * stride_wk + (k + offs_k)
        
        a = tl.load(input_ptr + offs_im, mask=mask_k, other=0.0)
        w = tl.load(weight_ptr + offs_wk, mask=mask_k, other=0.0)
        
        acc += tl.sum(a * w)
    
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + col)
    
    tl.store(output_ptr + row * stride_om + col, acc)


@torch.fx.wrap
def triton_linear(input_tensor, weight, bias=None):
    """Pure Triton linear."""
    B, S, I = input_tensor.shape
    O = weight.shape[0]
    
    # Assume inputs are already on GPU (they should be from the pattern)
    output = torch.empty((B, S, O), device='cuda', dtype=torch.float32)
    
    BLOCK_SIZE = 512
    grid = (B * S, O)
    
    triton_linear_kernel[grid](
        input_tensor, weight, bias, output,
        B, S, O,
        input_tensor.stride(0), input_tensor.stride(1), 
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE
    )
    return output


def replacement_fn(in_0, in_1, in_2, in_3):
    """Replacement function - fuse linear+reshape+split+permute+transpose."""
    B = in_3.shape[0]
    S = 49
    H = 8
    DQ = 32
    DV = 128
    
    linear_out = triton_linear(in_3, in_2, in_1)
    
    linear_out = linear_out.view(B, S, H, DQ * 2 + DV)
    Q = linear_out[:, :, :, :DQ]
    K = linear_out[:, :, :, DQ:DQ*2]
    V = linear_out[:, :, :, DQ*2:]
    
    Q = Q.permute(0, 2, 1, 3).contiguous()
    K = K.permute(0, 2, 1, 3).contiguous()
    V = V.permute(0, 2, 1, 3).contiguous()
    
    K_T = K.transpose(-2, -1).contiguous()
    
    in_0_cuda = in_0.cuda()
    
    return (Q, in_0_cuda, K_T, V)


def replacement_func():
    return replacement_fn