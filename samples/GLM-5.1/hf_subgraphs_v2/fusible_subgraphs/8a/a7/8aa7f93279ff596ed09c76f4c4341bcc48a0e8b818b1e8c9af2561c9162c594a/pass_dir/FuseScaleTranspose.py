import torch
import triton
import triton.language as tl


# ============ Triton Kernels ============

@triton.jit
def scale_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    scale_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    out = x * scale_val
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def transpose_2d_kernel(
    in_ptr,
    out_ptr,
    dim0_size,
    dim1_size,
    dim2_size,
    dim3_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_off < dim2_size
    mask_n = n_off < dim3_size

    batch_stride = dim1_size * dim2_size * dim3_size
    dim2_stride = dim3_size
    dim3_stride = 1

    # Output shape: [dim0, dim1, dim3, dim2] -> [70, 1, 32, 49]
    # Output strides: [1*32*49, 32*49, 49, 1]
    batch_stride_out = dim1_size * dim3_size * dim2_size
    dim2_stride_out = dim2_size  # stride for the "32" dimension = 49
    dim3_stride_out = 1           # stride for the "49" dimension = 1

    in_offsets = pid_batch * batch_stride + m_off * dim2_stride + n_off * dim3_stride
    out_offsets = pid_batch * batch_stride_out + n_off * dim2_stride_out + m_off * dim3_stride_out

    mask = mask_m & mask_n
    val = tl.load(in_ptr + in_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + out_offsets, val, mask=mask)


# ============ Kernel Wrappers ============

@torch.fx.wrap
def _triton_scale(x, scale_val):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    scale_kernel[(num_programs,)](
        in_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        scale_val=scale_val,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@torch.fx.wrap
def _triton_transpose(x):
    shape_in = x.shape
    dim0_size = shape_in[0]
    dim1_size = shape_in[1]
    dim2_size = shape_in[2]
    dim3_size = shape_in[3]

    out = torch.empty(dim0_size, dim1_size, dim3_size, dim2_size, dtype=x.dtype, device=x.device)

    BLOCK_M = 16
    BLOCK_N = 16
    num_m = (dim2_size + BLOCK_M - 1) // BLOCK_M
    num_n = (dim3_size + BLOCK_N - 1) // BLOCK_N

    transpose_2d_kernel[(num_m, num_n, dim0_size * dim1_size)](
        in_ptr=x,
        out_ptr=out,
        dim0_size=dim0_size,
        dim1_size=dim1_size,
        dim2_size=dim2_size,
        dim3_size=dim3_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out


# ============ Shared Dispatch Wrapper ============

@torch.fx.wrap
def shared_dispatch_wrapper(*args):
    # args = (input_tensor(s), route_string)
    route = args[-1]
    if route == "scale":
        x, scale_val = args[0], args[1]
        return _triton_scale(x, scale_val)
    elif route == "transpose":
        x = args[0]
        return _triton_transpose(x)
    else:
        raise RuntimeError(f"Unknown route: {route}")


# ============ Pass: Scale (in_1 * constant) ============

def pattern(in_1):
    tmp_0 = in_1 * 0.1767766952966369
    return tmp_0


def replacement_args(in_1):
    return (in_1, 0.1767766952966369, "scale")


def replacement_func():
    return shared_dispatch_wrapper