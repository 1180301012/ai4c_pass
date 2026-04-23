import torch
import triton
import triton.language as tl

# Pattern matching function - matches the computation pattern
def pattern(in_0, in_1, in_2):
    """
    Match the computation pattern: in_2 * in_1 + in_0, then unbind and permute
    """
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    tmp_3 = torch.unbind(tmp_2, dim=2)
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    tmp_6 = tmp_5.permute(0, 2, 1)
    return (tmp_6, tmp_4)


def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for the replacement"""
    return (in_0, in_1, in_2)


@triton.jit
def fused_mul_add_kernel(
    result_ptr,
    in_0_ptr, in_1_ptr, in_2_ptr,
    n_elements,
    dim_17, dim_128,
    stride_in_0_dim0,
    stride_in_1_dim2,
    stride_in_2_dim1,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for mul+add
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate indices for result [batch, 17, 2, 128]
    # Each thread handles one element in the [batch*17*128*2] = [batch*17*128] * 2
    batch_idx = offsets // (dim_17 * dim_128 * 2)
    rest = offsets % (dim_17 * dim_128 * 2)
    dim2_idx = rest // (dim_17 * dim_128)  # 0 or 1
    rest2 = rest % (dim_17 * dim_128)
    dim1_idx = rest2 // dim_128
    dim3_idx = rest2 % dim_128
    
    # in_2 [batch, 17, 1, 128]
    offset_in_2 = batch_idx * stride_in_2_dim1 + dim1_idx * stride_in_2_dim1 + dim3_idx
    
    # in_0 [2, 128] - dim2_idx selects row
    offset_in_0 = dim2_idx * stride_in_0_dim0 + dim3_idx
    
    # in_1 [1, 1, 2, 128] - dim2_idx selects along dim 2
    offset_in_1 = dim2_idx * stride_in_1_dim2 + dim3_idx
    
    # Load and compute
    in_0_val = tl.load(in_0_ptr + offset_in_0, mask=mask, other=0.0)
    in_1_val = tl.load(in_1_ptr + offset_in_1, mask=mask, other=0.0)
    in_2_val = tl.load(in_2_ptr + offset_in_2, mask=mask, other=0.0)
    
    result = in_2_val * in_1_val + in_0_val
    
    tl.store(result_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2):
    """Wrapper that launches the fused kernel for mul+add only"""
    batch_size, dim_17, dim_1, dim_128 = in_2.shape
    
    # Result shape [batch, 17, 2, 128]
    result = torch.empty((batch_size, dim_17, 2, dim_128), dtype=in_0.dtype, device=in_0.device)
    
    total_elements = batch_size * dim_17 * dim_128 * 2
    BLOCK_SIZE = 512
    num_programs = max(1, (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    stride_in_0_dim0 = in_0.stride(0)
    stride_in_1_dim2 = in_1.stride(2)
    stride_in_2_dim1 = in_2.stride(1)
    
    grid = (num_programs,)
    
    fused_mul_add_kernel[grid](
        result, in_0, in_1, in_2,
        total_elements,
        dim_17, dim_128,
        stride_in_0_dim0,
        stride_in_1_dim2,
        stride_in_2_dim1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Then do unbind and permute with torch
    unbind_result = torch.unbind(result, dim=2)
    tmp_4 = unbind_result[0]
    tmp_5 = unbind_result[1]
    tmp_6 = tmp_5.permute(0, 2, 1)
    
    return (tmp_6, tmp_4)


def replacement_func():
    """Return the replacement function"""
    return fused_kernel_wrapper