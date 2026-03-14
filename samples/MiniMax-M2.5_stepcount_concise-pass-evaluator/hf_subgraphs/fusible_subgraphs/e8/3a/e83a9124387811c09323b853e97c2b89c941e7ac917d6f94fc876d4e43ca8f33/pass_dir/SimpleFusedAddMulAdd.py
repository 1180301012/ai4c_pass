import torch
import triton
import triton.language as tl


@triton.jit
def add_mul_add_kernel(
    in_2_ptr, in_3_ptr, in_1_ptr, in_0_ptr, out_ptr,
    batch_stride_2, seq_stride_2, hidden_stride_2,
    batch_stride_3, seq_stride_3, hidden_stride_3,
    batch_stride_out, seq_stride_out, hidden_stride_out,
    batch_size: tl.constexpr, seq_len: tl.constexpr, hidden_dim: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused add-mul-add kernel with proper parallelization"""
    # Calculate global program id and offset
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate hidden offset (which element within hidden dim)
    hidden_offset = offsets % hidden_dim
    # Calculate flat index for batch/seq (for loading bias/weight)
    flat_idx = offsets // hidden_dim
    
    # Load weight and bias - need to calculate which batch/seq position
    # weight and bias are 1D, indexed by hidden_offset
    weight = tl.load(in_1_ptr + hidden_offset, mask=mask)
    bias = tl.load(in_0_ptr + hidden_offset, mask=mask)
    
    # Load in_2 and in_3 using computed indices
    # We need to compute the 3D index from flat_idx
    # flat_idx = batch_idx * seq_len + seq_idx
    seq_idx = flat_idx % seq_len
    batch_idx = flat_idx // seq_len
    
    # Calculate base indices for in_2 and in_3
    base_idx_2 = batch_idx * batch_stride_2 + seq_idx * seq_stride_2 + hidden_offset
    base_idx_3 = batch_idx * batch_stride_3 + seq_idx * seq_stride_3 + hidden_offset
    
    # Load values
    val_2 = tl.load(in_2_ptr + base_idx_2, mask=mask)
    val_3 = tl.load(in_3_ptr + base_idx_3, mask=mask)
    
    # Compute: (in_3 + in_2) * in_1 + in_0
    tmp = (val_3 + val_2) * weight + bias
    
    # Calculate output index
    out_base = batch_idx * batch_stride_out + seq_idx * seq_stride_out + hidden_offset
    tl.store(out_ptr + out_base, tmp, mask=mask)


@torch.fx.wrap
def add_mul_add_opt(a, b, c, d):
    """Fused add-mul-add: ((d + c) * b) + a
    a = in_0 (bias, 1D), b = in_1 (weight, 1D), c = in_2 (3D), d = in_3 (3D)
    Compute: ((in_3 + in_2) * in_1) + in_0
    """
    # Get shape of 3D tensor
    batch_size, seq_len, hidden_dim = c.shape
    
    # Total number of elements
    n_elements = batch_size * seq_len * hidden_dim
    
    # For small tensors, use PyTorch (less kernel overhead)
    # For larger tensors, use Triton kernel
    MIN_ELEMENTS_FOR_TRITON = 100000  # Only use kernel for larger tensors
    
    if n_elements < MIN_ELEMENTS_FOR_TRITON:
        # Use PyTorch for small tensors
        t1 = d + c
        t2 = t1 * b
        t3 = t2 + a
        return t3
    
    # Create output tensor with same shape as c
    out = torch.empty_like(c)
    
    # Get contiguous strides
    stride_c = c.stride()
    stride_d = d.stride()
    stride_out = out.stride()
    
    # Calculate grid - use total elements / BLOCK_SIZE
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    add_mul_add_kernel[(num_programs,)](
        c, d, b, a, out,
        stride_c[0], stride_c[1], stride_c[2],
        stride_d[0], stride_d[1], stride_d[2],
        stride_out[0], stride_out[1], stride_out[2],
        batch_size, seq_len, hidden_dim,
        n_elements,
        BLOCK_SIZE,
    )
    
    return out


def pattern(a, b, c, d):
    """Match: ((d + c) * b) + a  which is (in_3 + in_2) * in_1 + in_0"""
    t1 = d + c   # tmp_2 = in_3 + in_2
    t2 = t1 * b  # tmp_3 = tmp_2 * in_1
    t3 = t2 + a  # tmp_4 = tmp_3 + in_0
    return t3


def replacement_args(a, b, c, d):
    return (a, b, c, d)


def replacement_func():
    return add_mul_add_opt