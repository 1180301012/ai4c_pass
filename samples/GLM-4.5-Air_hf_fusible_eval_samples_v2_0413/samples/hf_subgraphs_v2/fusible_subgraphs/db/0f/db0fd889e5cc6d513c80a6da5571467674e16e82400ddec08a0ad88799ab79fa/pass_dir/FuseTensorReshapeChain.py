import torch
import triton
import triton.language as tl

def pattern(transposed_input):
    # Match the tensor reshape chain: transpose -> view -> flatten -> transpose
    # This pattern appears twice in the model for position embeddings
    viewed = transposed_input.view(1, 32, 15, 15)
    interpolated = torch.nn.functional.interpolate(viewed, size=(15, 15), mode='bicubic', align_corners=False)
    flattened = interpolated.flatten(2)
    result = flattened.transpose(1, 2)
    return result

def replacement_args(transposed_input):
    return (transposed_input,)

@triton.jit
def fuse_reshape_chain_kernel(
    input_ptr, output_ptr, 
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    output_stride_0, output_stride_1, output_stride_2,
    n_features, BLOCK_SIZE: tl.constexpr
):
    """Kernel that fuses transpose+view+flatten+transpose operations"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_features
    
    # Calculate input indices for the fused operation
    # Original: [1, 32, 15, 15] -> view -> [1, 32, 225] -> flatten -> [1, 225, 32] -> transpose
    # We can directly map from flattened output index to input index
    batch_idx = offsets // (225 * 32)
    feature_idx = (offsets % (225 * 32)) // 32
    seq_idx = offsets % 32
    
    # Calculate original input indices after transpose
    orig_seq_idx = feature_idx
    orig_feature_idx = seq_idx
    
    # Load data directly from transposed input
    input_offset = batch_idx * input_stride_0 + orig_seq_idx * input_stride_1 + orig_feature_idx * input_stride_2
    input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Store to output
    output_offset = batch_idx * output_stride_0 + feature_idx * output_stride_1 + seq_idx * output_stride_2
    tl.store(output_ptr + output_offset, input_val, mask=mask)

@torch.fx.wrap
def fused_reshape_chain(input_tensor):
    """Optimized version that fuses multiple reshape operations"""
    input_shape = input_tensor.shape
    output_shape = (input_shape[0], input_shape[1] * input_shape[2] * input_shape[3], input_shape[2])
    
    # Handle the special case where we can directly reshape
    if len(input_shape) == 3 and input_shape[0] == 1 and input_shape[1] == 32 and input_shape[2] == 225:
        # This is already in the right shape, just transpose
        return input_tensor.transpose(1, 2)
    
    # For general case, use fused kernel
    n_elements = output_shape[0] * output_shape[1] * output_shape[2]
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Get input strides
    input_strides = input_tensor.stride()
    output_strides = output.stride()
    
    fuse_reshape_chain_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        input_stride_0=input_strides[0], input_stride_1=input_strides[1], input_stride_2=input_strides[2],
        output_stride_0=output_strides[0], output_stride_1=output_strides[1], output_stride_2=output_strides[2],
        n_features=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_reshape_chain