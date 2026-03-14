import torch
import triton
import triton.language as tl

# Pattern for start1037_end1041_48 graphs
# interpolate(in_0, (16,12)) * in_2, interpolate(in_1, (8,6)) * in_3

def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = torch.nn.functional.interpolate(in_0, [16, 12], None, 'nearest')
    tmp_1 = in_2 * tmp_0
    tmp_2 = torch.nn.functional.interpolate(in_1, [8, 6], None, 'nearest')
    tmp_3 = in_3 * tmp_2
    return (tmp_1, tmp_3)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_upsample_mul_kernel_2x_c(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    N, C, H_out, W_out,
    H_in, W_in,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused nearest-neighbor 2x upsample + element-wise multiply kernel"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Convert flat index to NCHW indices for output
    w_out = offsets % W_out
    tmp = offsets // W_out
    h_out = tmp % H_out
    tmp = tmp // H_out
    c = tmp % C
    n = tmp // C
    
    # Nearest neighbor: compute input indices (2x upscale)
    h_in = h_out >> 1  # h_out // 2
    w_in = w_out >> 1  # w_out // 2
    
    # Compute flat input index
    input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in
    
    # Load, multiply, store
    inp_val = tl.load(input_ptr + input_idx, mask=mask)
    other_val = tl.load(other_ptr + offsets, mask=mask)
    result = inp_val * other_val
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_identity_mul_kernel_c(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused identity (1x) interpolation + element-wise multiply kernel (just multiply)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load, multiply, store
    inp_val = tl.load(input_ptr + offsets, mask=mask)
    other_val = tl.load(other_ptr + offsets, mask=mask)
    result = inp_val * other_val
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_interpolate_mul_impl_c(in_0, in_1, in_2, in_3):
    # First operation: interpolate(in_0, (16, 12)) * in_2
    # in_0: [N, C, 8, 6] -> [N, C, 16, 12]
    N, C, H_in1, W_in1 = in_0.shape
    H_out1, W_out1 = 16, 12
    n_elements1 = N * C * H_out1 * W_out1
    
    out1 = torch.empty(N, C, H_out1, W_out1, device=in_0.device, dtype=in_0.dtype)
    
    grid1 = lambda meta: (triton.cdiv(n_elements1, meta['BLOCK_SIZE']),)
    fused_upsample_mul_kernel_2x_c[grid1](
        in_0, in_2, out1,
        n_elements1,
        N, C, H_out1, W_out1,
        H_in1, W_in1,
    )
    
    # Second operation: interpolate(in_1, (8, 6)) * in_3
    # in_1: [N, C2, 8, 6] -> [N, C2, 8, 6] (identity)
    N2, C2, H2, W2 = in_1.shape
    n_elements2 = N2 * C2 * H2 * W2
    
    out2 = torch.empty_like(in_3)
    
    grid2 = lambda meta: (triton.cdiv(n_elements2, meta['BLOCK_SIZE']),)
    fused_identity_mul_kernel_c[grid2](
        in_1, in_3, out2,
        n_elements2,
    )
    
    return (out1, out2)


def replacement_func():
    return fused_interpolate_mul_impl_c