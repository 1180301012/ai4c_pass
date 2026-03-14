import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Match the transpose pattern: in_0.transpose(-1, -2)
    """
    return in_0.transpose(-1, -2)


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def transpose_kernel(
    in_ptr, out_ptr,
    M, N,
    stride_in_0, stride_in_1, stride_in_2, stride_in_3,
    stride_out_0, stride_out_1, stride_out_2, stride_out_3,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # Original shape: [..., M, N], transposed: [..., N, M]
    # For 4D tensors: [B, H, M, N] -> [B, H, N, M]
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    
    group_id = pid // num_pid_in_group
    pid_in_group = pid % num_pid_in_group
    
    if group_id == 0:
        # Transpose of last two dimensions for 4D tensor
        pid_m = pid_in_group // num_pid_n
        pid_n = pid_in_group % num_pid_n
        
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        
        # Input: [B, H, M, N], output: [B, H, N, M]
        # Load from in (using first two dims as batch/head)
        ptrs = in_ptr + offs_m[None, :] * stride_in_2 + offs_n[:, None] * stride_in_3
        mask = (offs_m[None, :] < M) & (offs_n[:, None] < N)
        
        x = tl.load(ptrs, mask=mask, other=0.0)
        
        # Store transposed: output uses stride_out_2 for M dim and stride_out_3 for N dim
        out_ptrs = out_ptr + offs_n[None, :] * stride_out_2 + offs_m[:, None] * stride_out_3
        tl.store(out_ptrs, x, mask=mask)


@torch.fx.wrap
def transpose_wrapper(in_0):
    # For 4D tensor [B, H, M, N] -> [B, H, N, M]
    M = in_0.shape[-2]
    N = in_0.shape[-1]
    
    # Output shape: swap last two dimensions
    out_shape = list(in_0.shape)
    out_shape[-2], out_shape[-1] = out_shape[-1], out_shape[-2]
    out = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    
    grid = lambda M, N: (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
    
    transpose_kernel[grid(M, N)](
        in_0, out,
        M, N,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    
    return out


def replacement_func():
    return transpose_wrapper