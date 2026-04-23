import torch
import triton
import triton.language as tl


# Autotune configs for optimal performance across different tensor sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256, 'num_warps': 2, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE': 512, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE': 2048, 'num_warps': 8, 'num_stages': 3}),
    ],
    key=['N'],
)
@triton.jit
def triton_permute_kernel(input_ptr, output_ptr, B, S, H, D, N,
                          input_stride_b, input_stride_s, input_stride_h, input_stride_d,
                          output_stride_b, output_stride_h, output_stride_s, output_stride_d,
                          BLOCK_SIZE: tl.constexpr):
    """Optimized permute kernel with stride-aware memory access and autotuning"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    total_elements = N
    mask = offsets < total_elements
    
    # Compute b, s, h, d indices from flat offset
    tmp = offsets
    b = tmp // (S * H * D)
    tmp = tmp % (S * H * D)
    s = tmp // (H * D)
    tmp = tmp % (H * D)
    h = tmp // D
    d = tmp % D
    
    # Use actual strides for input and output
    input_offset = (b * input_stride_b + s * input_stride_s + 
                    h * input_stride_h + d * input_stride_d)
    
    output_offset = (b * output_stride_b + h * output_stride_h + 
                     s * output_stride_s + d * output_stride_d)
    
    val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + output_offset, val, mask=mask)


def pattern(x):
    """Match a single permute(0, 2, 1, 3) operation"""
    return x.permute(0, 2, 1, 3)


def replacement_args(x):
    return (x,)


@torch.fx.wrap
def triton_permute_func(x):
    """Triton-based permute(0, 2, 1, 3) with stride awareness and autotuning"""
    B, S, H, D = x.shape
    N = B * H * S * D
    
    # Allocate output
    out = torch.empty(B, H, S, D, dtype=x.dtype, device=x.device)
    
    # Get actual strides
    input_strides = x.stride()
    output_strides = out.stride()
    
    # Launch kernel with autotuning
    grid = ((N + 255) // 256,)
    
    triton_permute_kernel[grid](
        x, out, B, S, H, D, N,
        input_strides[0], input_strides[1], input_strides[2], input_strides[3],
        output_strides[0], output_strides[1], output_strides[2], output_strides[3],
    )
    
    return out


def replacement_func():
    return triton_permute_func