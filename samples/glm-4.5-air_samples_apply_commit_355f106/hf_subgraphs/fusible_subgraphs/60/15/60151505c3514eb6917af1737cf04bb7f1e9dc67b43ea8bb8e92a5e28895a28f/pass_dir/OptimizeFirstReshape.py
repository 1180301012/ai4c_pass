import torch
import triton
import triton.language as tl

def pattern(x):
    # Simple pattern: match just a single reshape operation
    tmp = x.reshape(-1, 256, -1)
    return tmp

def replacement_args(x):
    return (x,)

@triton.jit
def reshape_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    features,
    spatial_size,
    src_stride_0,
    src_stride_1,
    src_stride_2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // spatial_size
    spatial_idx = pid % spatial_size
    
    if batch_idx >= batch_size or spatial_idx >= spatial_size:
        return
    
    # Input offset: batch_idx * src_stride_0 + spatial_idx * src_stride_2
    # Output offset: batch_idx * features * spatial_size + spatial_idx * features
    src_offset = batch_idx * src_stride_0 + spatial_idx * src_stride_2
    dst_offset = batch_idx * features * spatial_size + spatial_idx * features
    
    # Load a full row from source (features dimension)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < features
    src_ptr = input_ptr + src_offset
    dst_ptr = output_ptr + dst_offset
    
    # Load and store the feature dimension
    values = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, values, mask=mask)

@torch.fx.wrap
def optimized_reshape_first(in_tensor, target_batch, features, target_spatial):
    total_elements = target_batch * target_spatial
    BLOCK_SIZE = 256
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output_shape = (target_batch, features, target_spatial)
    out = torch.empty(output_shape, dtype=in_tensor.dtype, device=in_tensor.device)
    
    # Get input strides for efficient memory access
    src_stride_0 = in_tensor.stride(0)
    src_stride_1 = in_tensor.stride(1)
    src_stride_2 = in_tensor.stride(2)
    
    reshape_kernel[(num_programs,)](
        input_ptr=in_tensor,
        output_ptr=out,
        batch_size=target_batch,
        features=features,
        spatial_size=target_spatial,
        src_stride_0=src_stride_0,
        src_stride_1=src_stride_1,
        src_stride_2=src_stride_2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    def kernel_wrapper(x):
        # Simple optimized reshape for a single tensor
        batch_size, features, spatial_size, _ = x.shape
        tmp = optimized_reshape_first(x, batch_size, features, spatial_size)
        return tmp
    
    return kernel_wrapper