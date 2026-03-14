import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """Pattern: flatten(2) followed by transpose(1, 2)"""
    tmp = input_tensor.flatten(2)
    result = tmp.transpose(1, 2)
    return result

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def fused_flatten_transpose_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    dim1,
    dim2,
    dim3,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    in_stride_0,
    in_stride_1,
    in_stride_2,
    in_stride_3,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for flatten(2) + transpose(1, 2)
    Input: [B, C, H, W]
    After flatten(2): [B, C, H*W]
    After transpose(1, 2): [B, H*W, C]
    """
    pid = tl.program_id(0)
    num_elements = batch_size * dim1 * dim2 * dim3
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Calculate input indices (B, C, H, W)
    b = offsets // (dim1 * dim2 * dim3)
    remainder = offsets % (dim1 * dim2 * dim3)
    c = remainder // (dim2 * dim3)
    remainder = remainder % (dim2 * dim3)
    h = remainder // dim3
    w = remainder % dim3
    
    # Load from input using strides
    in_offsets = b * in_stride_0 + c * in_stride_1 + h * in_stride_2 + w * in_stride_3
    data = tl.load(input_ptr + in_offsets, mask=mask, other=0.0)
    
    # Calculate output indices: [B, H*W, C]
    # After flatten: spatial_idx = h * W + w, so position is [b, c, spatial_idx]
    # After transpose: position is [b, spatial_idx, c]
    spatial_idx = h * dim3 + w
    out_offsets = b * out_stride_0 + spatial_idx * out_stride_1 + c * out_stride_2
    
    tl.store(output_ptr + out_offsets, data, mask=mask)

@torch.fx.wrap
def fused_flatten_transpose(input_tensor):
    """Wrapper for fused flatten + transpose operation"""
    batch_size, dim1, dim2, dim3 = input_tensor.shape
    
    # Output shape: [B, H*W, C] = [batch_size, dim2*dim3, dim1]
    output = torch.empty(batch_size, dim2 * dim3, dim1, 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    num_elements = batch_size * dim1 * dim2 * dim3
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    
    fused_flatten_transpose_kernel[grid](
        input_tensor,
        output,
        batch_size,
        dim1,
        dim2,
        dim3,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        input_tensor.stride(0),
        input_tensor.stride(1),
        input_tensor.stride(2),
        input_tensor.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_flatten_transpose