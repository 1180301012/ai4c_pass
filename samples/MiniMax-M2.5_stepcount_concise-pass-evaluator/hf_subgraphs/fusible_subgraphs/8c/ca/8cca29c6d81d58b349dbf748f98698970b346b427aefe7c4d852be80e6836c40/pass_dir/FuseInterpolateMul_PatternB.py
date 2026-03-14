import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16}, num_stages=4, num_warps=1),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=8),
    ],
    key=['B', 'C', 'H_out', 'W_out'],
)
@triton.jit
def interpolate_mul_kernel(
    input_ptr, other_ptr, output_ptr,
    B, C, H_in, W_in, H_out, W_out,
    scale_h, scale_w,
    stride_input_b, stride_input_c, stride_input_h, stride_input_w,
    stride_other_b, stride_other_c, stride_other_h, stride_other_w,
    stride_output_b, stride_output_c, stride_output_h, stride_output_w,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused interpolate (nearest neighbor) + multiply kernel.
    
    For nearest neighbor interpolation:
    output[b, c, h, w] = input[b, c, h * scale_h, w * scale_w] * other[b, c, h, w]
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(H_out, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(W_out, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, H_out - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_b = tl.program_id(1)
    offs_c = tl.arange(0, C)
    
    mask_m = offs_m < H_out
    mask_n = offs_n < W_out
    
    other_ptrs = (
        other_ptr
        + offs_b * stride_other_b
        + offs_c[:, None, None] * stride_other_c
        + offs_m[None, :, None] * stride_other_h
        + offs_n[None, None, :] * stride_other_w
    )
    
    input_offs_h = (offs_m * scale_h).to(tl.int64)
    input_offs_w = (offs_n * scale_w).to(tl.int64)
    
    input_offs_h = tl.minimum(tl.maximum(input_offs_h, 0), H_in - 1)
    input_offs_w = tl.minimum(tl.maximum(input_offs_w, 0), W_in - 1)
    
    input_ptrs = (
        input_ptr
        + offs_b * stride_input_b
        + offs_c[:, None, None] * stride_input_c
        + input_offs_h[None, :, None] * stride_input_h
        + input_offs_w[None, None, :] * stride_input_w
    )
    
    for c in range(0, C):
        input_val = tl.load(
            input_ptrs + c * stride_input_c,
            mask=mask_m[None, :, None] & mask_n[None, None, :],
            other=0.0
        )
        
        other_val = tl.load(
            other_ptrs + c * stride_other_c,
            mask=mask_m[None, :, None] & mask_n[None, None, :],
            other=0.0
        )
        
        result = input_val * other_val
        
        output_ptrs = (
            output_ptr
            + offs_b * stride_output_b
            + c * stride_output_c
            + offs_m[:, None] * stride_output_h
            + offs_n[None, :] * stride_output_w
        )
        tl.store(output_ptrs, result, mask=mask_m[:, None] & mask_n[None, :])


def interpolate_mul_triton(input_tensor, other_tensor, size_h, size_w):
    B, C, H_in, W_in = input_tensor.shape
    _, _, H_out, W_out = other_tensor.shape
    
    scale_h = H_in / H_out
    scale_w = W_in / W_out
    
    output = torch.empty((B, C, H_out, W_out), device=input_tensor.device, dtype=input_tensor.dtype)
    
    grid = (H_out * W_out, B)
    
    interpolate_mul_kernel[grid](
        input_tensor, other_tensor, output,
        B, C, H_in, W_in, H_out, W_out,
        scale_h, scale_w,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        other_tensor.stride(0), other_tensor.stride(1), other_tensor.stride(2), other_tensor.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    
    return output


@torch.fx.wrap
def interpolate_mul_wrapper(input_tensor, other_tensor, size_h, size_w):
    return interpolate_mul_triton(input_tensor, other_tensor, size_h, size_w)


def pattern(in_0, in_1, in_2, in_3):
    """
    Match Pattern B: interpolate(in_1, (32, 24)) * in_2, interpolate(in_0, (16, 12)) * in_3
    (swapped input order compared to Pattern A)
    """
    tmp_0 = torch.nn.functional.interpolate(in_1, size=(32, 24), mode='nearest')
    tmp_1 = in_2 * tmp_0
    
    tmp_2 = torch.nn.functional.interpolate(in_0, size=(16, 12), mode='nearest')
    tmp_3 = in_3 * tmp_2
    
    return tmp_1, tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    # First branch: interpolate in_1 to (32, 24), multiply with in_2
    # Second branch: interpolate in_0 to (16, 12), multiply with in_3
    return (in_1, in_2, 32, 24, in_0, in_3, 16, 12)


def replacement_func():
    return interpolate_mul_wrapper