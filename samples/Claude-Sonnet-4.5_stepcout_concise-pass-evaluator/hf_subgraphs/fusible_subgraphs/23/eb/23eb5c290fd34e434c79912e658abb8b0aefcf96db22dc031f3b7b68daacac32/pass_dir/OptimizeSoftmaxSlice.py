import torch
import triton
import triton.language as tl


def pattern(input_tensor):
    """
    Match softmax + slice pattern
    """
    tmp_3 = torch.nn.functional.softmax(input_tensor, dim=-1)
    tmp_4 = tmp_3[Ellipsis, slice(None, 64, None)]
    return (tmp_3, tmp_4)


def replacement_args(input_tensor):
    return (input_tensor,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def softmax_slice_kernel(
    input_ptr, output_full_ptr, output_slice_ptr,
    B, H, W, N,
    stride_b, stride_h, stride_w, stride_n,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_h = pid_hw // W
    pid_w = pid_hw % W
    
    base_offset = pid_b * stride_b + pid_h * stride_h + pid_w * stride_w
    
    # Load data
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    data = tl.load(input_ptr + base_offset + offsets * stride_n, mask=mask, other=float('-inf'))
    
    # Softmax
    max_val = tl.max(data, axis=0)
    exp_data = tl.exp(data - max_val)
    sum_exp = tl.sum(exp_data, axis=0)
    softmax_out = exp_data / sum_exp
    
    # Store full output
    tl.store(output_full_ptr + base_offset + offsets * stride_n, softmax_out, mask=mask)
    
    # Store sliced output (first 64)
    slice_mask = offsets < 64
    tl.store(output_slice_ptr + base_offset + offsets * stride_n, softmax_out, mask=slice_mask)


@torch.fx.wrap
def softmax_slice_impl(input_tensor):
    B, H, W, N = input_tensor.shape
    
    output_full = torch.empty_like(input_tensor)
    output_slice = torch.empty((B, H, W, 64), device=input_tensor.device, dtype=input_tensor.dtype)
    
    grid = (B, H * W)
    
    softmax_slice_kernel[grid](
        input_tensor, output_full, output_slice,
        B, H, W, N,
        input_tensor.stride(0), input_tensor.stride(1), 
        input_tensor.stride(2), input_tensor.stride(3),
    )
    
    return output_full, output_slice


def replacement_func():
    return softmax_slice_impl