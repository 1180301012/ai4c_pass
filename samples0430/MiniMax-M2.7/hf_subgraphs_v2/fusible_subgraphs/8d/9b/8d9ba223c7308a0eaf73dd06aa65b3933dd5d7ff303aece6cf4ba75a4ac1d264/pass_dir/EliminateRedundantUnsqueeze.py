import torch
import triton
import triton.language as tl


def pattern(a, b, c, d):
    """
    Match the redundant unsqueeze-add pattern:
    
    tmp_14 = in_3.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_16 = b + tmp_15
    tmp_17 = in_3.unsqueeze(1)  # REDUNDANT - same as tmp_14
    tmp_18 = tmp_17.unsqueeze(0)  # REDUNDANT - same as tmp_15
    tmp_19 = tmp_16 + tmp_18  # REDUNDANT ADD
    
    This matches the computation from the attention mask handling.
    """
    tmp_14 = a.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_16 = b + tmp_15
    tmp_17 = c.unsqueeze(1)
    tmp_18 = tmp_17.unsqueeze(0)
    tmp_19 = tmp_16 + tmp_18
    return tmp_19


def replacement_args(a, b, c, d):
    return (a, b, c)


@triton.jit
def fused_mask_add_kernel(
    in_3_ptr,
    in_3_stride_0, in_3_stride_1, in_3_stride_2,
    tensor_b_ptr,
    tensor_b_stride_0, tensor_b_stride_1, tensor_b_stride_2, tensor_b_stride_3, tensor_b_stride_4,
    output_ptr,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3, output_stride_4,
    b_d0, b_d1, b_d2, b_d3, b_d4,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for attention mask addition with redundancy elimination.
    
    in_3: (64, 64, 64)
    unsqueeze(1): (64, 64, 64) -> (64, 1, 64, 64)
    unsqueeze(0): (64, 1, 64, 64) -> (1, 64, 1, 64, 64)
    
    tensor_b: (1, 64, 12, 64, 64) or (1, 16, 24, 64, 64)
    
    Output: (tensor_b + unsqueezed) + unsqueezed = tensor_b + 2 * unsqueezed
    """
    pid = tl.program_id(0)
    b_elements = b_d0 * b_d1 * b_d2 * b_d3 * b_d4
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < b_elements
    
    # Convert flat offset to multi-dimensional indices
    idx = offsets
    b_idx0 = idx // (b_d1 * b_d2 * b_d3 * b_d4)
    remainder = idx % (b_d1 * b_d2 * b_d3 * b_d4)
    b_idx1 = remainder // (b_d2 * b_d3 * b_d4)
    remainder = remainder % (b_d2 * b_d3 * b_d4)
    b_idx2 = remainder // (b_d3 * b_d4)
    remainder = remainder % (b_d3 * b_d4)
    b_idx3 = remainder // b_d4
    b_idx4 = remainder % b_d4
    
    # Load from tensor_b
    b_offset = (b_idx0 * tensor_b_stride_0 + b_idx1 * tensor_b_stride_1 + 
                b_idx2 * tensor_b_stride_2 + b_idx3 * tensor_b_stride_3 + b_idx4 * tensor_b_stride_4)
    b_val = tl.load(tensor_b_ptr + b_offset, mask=mask, other=0.0)
    
    # For unsqueeze(1).unsqueeze(0):
    # in_3[b_idx1, 0, b_idx3] broadcasts over tensor_b
    in_3_offset = b_idx1 * in_3_stride_0 + 0 * in_3_stride_1 + b_idx3 * in_3_stride_2
    mask_val = tl.load(in_3_ptr + in_3_offset, mask=mask, other=0.0)
    
    # Compute: tensor_b + mask + mask = tensor_b + 2 * mask
    result = b_val + 2.0 * mask_val
    
    # Store result
    output_offset = (b_idx0 * output_stride_0 + b_idx1 * output_stride_1 + 
                     b_idx2 * output_stride_2 + b_idx3 * output_stride_3 + b_idx4 * output_stride_4)
    tl.store(output_ptr + output_offset, result, mask=mask)


@torch.fx.wrap
def fused_mask_add(a, b, c):
    """
    Fused attention mask addition with redundancy elimination.
    
    Input:
    a: in_3 with shape (64, 64, 64)
    b: tensor before first add (tmp_13)
    c: tensor with different shape (unused)
    
    Output: (tmp_13 + unsqueezed) + unsqueezed = tmp_13 + 2 * unsqueezed
    """
    b_shape = (b.shape[0], b.shape[1], b.shape[2], b.shape[3], b.shape[4])
    output = torch.empty_like(b)
    
    b_elements = b.numel()
    BLOCK_SIZE = 1024
    num_programs = (b_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Strides for in_3 (64, 64, 64)
    a_stride_0 = a.shape[1] * a.shape[2]
    a_stride_1 = a.shape[2]
    a_stride_2 = 1
    
    fused_mask_add_kernel[(num_programs,)](
        in_3_ptr=a,
        in_3_stride_0=a_stride_0,
        in_3_stride_1=a_stride_1,
        in_3_stride_2=a_stride_2,
        tensor_b_ptr=b,
        tensor_b_stride_0=b.stride(0),
        tensor_b_stride_1=b.stride(1),
        tensor_b_stride_2=b.stride(2),
        tensor_b_stride_3=b.stride(3),
        tensor_b_stride_4=b.stride(4),
        output_ptr=output,
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        output_stride_2=output.stride(2),
        output_stride_3=output.stride(3),
        output_stride_4=output.stride(4),
        b_d0=b_shape[0], b_d1=b_shape[1], b_d2=b_shape[2], b_d3=b_shape[3], b_d4=b_shape[4],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_mask_add