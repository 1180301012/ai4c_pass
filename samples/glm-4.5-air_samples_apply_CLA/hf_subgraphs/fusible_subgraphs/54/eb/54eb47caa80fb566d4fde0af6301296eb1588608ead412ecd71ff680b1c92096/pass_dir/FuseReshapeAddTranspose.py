import torch
import triton
import triton.language as tl

# Pattern: match reshape + add + transpose
# This uses both inputs so there's no dead code
def pattern(a, b):
    # a = in_0, b = in_1
    t = b.reshape(1, 64, 256)
    c = a + t
    d = c.transpose(0, 1)
    return d

def replacement_args(a, b):
    return (a, b)

# Simple optimized transpose kernel
@triton.jit
def transpose_kernel(
    in_ptr, out_ptr,
    dim0, dim1, dim2,
    stride_in_0, stride_in_1, stride_in_2,
    stride_out_0, stride_out_1, stride_out_2,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program computes one element
    pid = tl.program_id(0)
    total_elements = dim0 * dim1 * dim2
    
    if pid >= total_elements:
        return
    
    # Calculate 3D indices
    idx = pid
    d2 = idx % dim2
    idx = idx // dim2
    d1 = idx % dim1
    d0 = idx // dim1
    
    # For transpose(0,1), output[d1, d0, d2] = input[d0, d1, d2]
    in_offset = d0 * stride_in_0 + d1 * stride_in_1 + d2 * stride_in_2
    out_offset = d1 * stride_out_0 + d0 * stride_out_1 + d2 * stride_out_2
    
    val = tl.load(in_ptr + in_offset)
    tl.store(out_ptr + out_offset, val)


@torch.fx.wrap
def transpose_kernel_wrapper(a):
    # Input shape: [dim0, dim1, dim2]
    # Output shape: [dim1, dim0, dim2]
    dim0, dim1, dim2 = a.shape
    
    out = torch.empty((dim1, dim0, dim2), dtype=a.dtype, device=a.device)
    
    total_elements = dim0 * dim1 * dim2
    BLOCK_SIZE = 1024
    
    transpose_kernel[(total_elements,)](
        a, out,
        dim0, dim1, dim2,
        a.stride(0), a.stride(1), a.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return fused_kernel_wrapper


# Optimized kernel for reshape + add + transpose
@triton.jit
def fused_kernel(
    a_ptr, b_ptr, out_ptr,
    dim0, dim1, dim2,
    stride_a0, stride_a1, stride_a2,
    stride_b0, stride_b1, stride_b2,
    stride_out0, stride_out1, stride_out2,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program computes one row of output
    pid = tl.program_id(0)
    num_rows = dim0 * dim2
    
    if pid >= num_rows:
        return
    
    row = pid % dim0
    plane = pid // dim0
    
    # Load a[row, plane, :]
    mask = tl.arange(0, dim2) < dim2
    a_offset = row * stride_a0 + plane * stride_a2 + tl.arange(0, dim2)
    a_vals = tl.load(a_ptr + a_offset, mask=mask, other=0.0)
    
    # Load b - reshape [64, 2, 128] -> [1, 64, 256]
    # b[plane, :] corresponds to the reshaped view
    # Original stride_b0 = 2*128 = 256
    b_offset = plane * stride_b0 + tl.arange(0, dim2)
    b_vals = tl.load(b_ptr + b_offset, mask=mask, other=0.0)
    
    # Add
    result = a_vals + b_vals
    
    # Store to output (transposed: out[plane, row, :] = result)
    out_offset = plane * stride_out0 + row * stride_out2 + tl.arange(0, dim2)
    tl.store(out_ptr + out_offset, result, mask=mask)


@torch.fx.wrap
def fused_kernel_wrapper(a, b):
    dim0, dim1, dim2 = a.shape
    
    out = torch.empty((dim1, dim0, dim2), dtype=a.dtype, device=a.device)
    
    num_rows = dim0 * dim2
    BLOCK_SIZE = min(dim2, 1024)
    
    fused_kernel[(num_rows,)](
        a, b, out,
        dim0, dim1, dim2,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_SIZE,
    )
    
    return out