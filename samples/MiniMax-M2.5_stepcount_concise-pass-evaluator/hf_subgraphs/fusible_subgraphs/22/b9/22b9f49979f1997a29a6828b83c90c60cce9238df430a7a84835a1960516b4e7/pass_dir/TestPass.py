import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """Simple test pattern to debug matching."""
    # Test with exact model.py structure
    cat_result = torch.cat([in_0, in_1, in_2], dim=1)
    reshape_result = cat_result.reshape(1, 8, 40, 576)
    transpose_result = reshape_result.transpose(-1, -2)
    mul_result = in_3 * transpose_result
    pad_result = torch.nn.functional.pad(mul_result, (0, 0, 1, 0, 0, 0), 'constant', None)
    return (pad_result,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
    ],
    key=['N', 'K'],
)
@triton.jit
def fused_kernel_debug(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, out_ptr,
    N, K, M,
    stride_in0_0, stride_in0_1, stride_in0_2, stride_in0_3,
    stride_in1_0, stride_in1_1, stride_in1_2, stride_in1_3,
    stride_in2_0, stride_in2_1, stride_in2_2, stride_in2_3,
    stride_in3_0, stride_in3_1, stride_in3_2, stride_in3_3,
    stride_out_0, stride_out_1, stride_out_2, stride_out_3,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    
    pid_n = pid % num_pid_n
    pid_k = pid // num_pid_n
    
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    mask_n = offs_n < N
    mask_k = offs_k < K
    
    out_offs = (1 * stride_out_1 + offs_n[:, None] * stride_out_2 + offs_k[None, :] * stride_out_3)
    
    C0, C1, C2 = 80, 120, 120
    head_dim = 40
    
    for head in range(8):
        in3_offs = (0 * stride_in3_0 + head * stride_in3_1 + offs_n[:, None] * stride_in3_2 + offs_k[None, :] * stride_in3_3)
        in3_ptrs = in_3_ptr + in3_offs
        in3 = tl.load(in3_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        result = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
        
        head_start = head * head_dim
        head_end = (head + 1) * head_dim
        
        if head_start < C0:
            in0_start = head_start
            in0_end = min(head_end, C0)
            for ch in range(in0_end - in0_start):
                ch_idx = in0_start + ch
                in0_offs = (0 * stride_in0_0 + ch_idx * stride_in0_1 + offs_n[:, None] * stride_in0_2 + ch * stride_in0_3)
                in0_ptrs = in_0_ptr + in0_offs
                in0_val = tl.load(in0_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
                result += in0_val
        
        if head_start >= C0 and head_start < C0 + C1:
            in1_start = head_start - C0
            in1_end = min(head_end - C0, C1)
            for ch in range(in1_end - in1_start):
                ch_idx = C0 + in1_start + ch
                in1_offs = (0 * stride_in1_0 + in1_start * stride_in1_1 + offs_n[:, None] * stride_in1_2 + ch * stride_in1_3)
                in1_ptrs = in_1_ptr + in1_offs
                in1_val = tl.load(in1_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
                result += in1_val
        
        if head_start >= C0 + C1:
            in2_start = head_start - C0 - C1
            in2_end = min(head_end - C0 - C1, C2)
            for ch in range(in2_end - in2_start):
                ch_idx = C0 + C1 + in2_start + ch
                in2_offs = (0 * stride_in2_0 + in2_start * stride_in2_1 + offs_n[:, None] * stride_in2_2 + ch * stride_in2_3)
                in2_ptrs = in_2_ptr + in2_offs
                in2_val = tl.load(in2_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
                result += in2_val
        
        out_val = result * in3
        out_ptrs = out_ptr + out_offs
        tl.store(out_ptrs, out_val, mask=mask_n[:, None] & mask_k[None, :])


@torch.fx.wrap
def fused_kernel_wrapper_debug(in_0, in_1, in_2, in_3):
    N = in_3.shape[2]
    K = in_3.shape[3]
    
    output = torch.zeros((1, 9, N, K), dtype=torch.float32, device=in_0.device)
    
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_k = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    grid = (num_pid_n * num_pid_k,)
    
    fused_kernel_debug[grid](
        in_0, in_1, in_2, in_3, output,
        N, K, 8,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    
    return output


def replacement_func():
    return fused_kernel_wrapper_debug