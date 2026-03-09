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
def reshape_permute_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    features,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // spatial_size
    spatial_idx = pid % spatial_size
    
    if batch_idx >= batch_size or spatial_idx >= spatial_size:
        return
    
    src_offset = batch_idx * features * spatial_size + spatial_idx
    dst_offset = batch_idx * spatial_size * features + spatial_idx * features
    
    # Load a full row from source (features dimension)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < features
    src_ptr = input_ptr + src_offset * features
    dst_ptr = output_ptr + dst_offset
    
    # Load and store with appropriate strides
    values = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets * spatial_size, values, mask=mask)

@torch.fx.wrap
def optimized_reshape_permute(in_tensor, batch_size, features, spatial_size):
    total_elements = batch_size * spatial_size
    BLOCK_SIZE = 256
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output_shape = (batch_size, spatial_size, features)
    out = torch.empty(output_shape, dtype=in_tensor.dtype, device=in_tensor.device)
    
    reshape_permute_kernel[(num_programs,)](
        input_ptr=in_tensor,
        output_ptr=out,
        batch_size=batch_size,
        features=features,
        spatial_size=spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    def kernel_wrapper(x):
        # Simple optimized reshape for a single tensor
        batch_size, features, spatial_size, _ = x.shape
        tmp = optimized_reshape_permute(x, batch_size, features, spatial_size)
        return tmp
    
    return kernel_wrapper