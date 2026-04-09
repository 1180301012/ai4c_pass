import torch
import triton
import triton.language as tl

def pattern(conv3d):
    """
    Match the flatten + transpose sequence that can be fused
    tmp_7 = conv3d.flatten(2)
    tmp_8 = tmp_7.transpose(1, 2)
    """
    tmp_7 = conv3d.flatten(2)
    tmp_8 = tmp_7.transpose(1, 2)
    return tmp_8

def replacement_args(conv3d):
    return (conv3d,)

@triton.jit
def fused_flatten_transpose_kernel(
    input_ptr,
    output_ptr,
    input_dims0, input_dims1, input_dims2, input_dims3, input_dims4,
    output_dims0, output_dims1, output_dims2,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that fuses flatten(2) + transpose(1, 2) operations
    """
    pid = tl.program_id(0)
    n_elements = output_dims0 * output_dims1 * output_dims2
    block_size = BLOCK_SIZE
    
    # Compute total blocks needed
    num_blocks = (n_elements + block_size - 1) // block_size
    if pid >= num_blocks:
        return
    
    # Each block handles a range of elements
    start_idx = pid * block_size
    end_idx = min(start_idx + block_size, n_elements)
    offsets = tl.arange(start_idx, end_idx)
    mask = offsets < n_elements
    
    # Calculate output indices for each element
    # Output shape: [original_dims0, original_dims3, original_dims1*original_dims2*original_dims4]
    idx1 = offsets // (output_dims1 * output_dims2)
    remainder = offsets % (output_dims1 * output_dims2)
    idx2 = remainder // output_dims2
    idx3 = remainder % output_dims2
    
    # Map output indices to input indices
    # Original flatten(2): keeps first 2 dims, flattens from dim 2
    # Original input shape: [dim0, dim1, dim2, dim3, dim4]
    # After flatten(2): [dim0, dim1, dim3, dim2*dim4]
    # After transpose(1, 2): [dim0, dim3, dim1*dim2*dim4]
    input_idx0 = idx1
    input_idx1 = idx2 // (input_dims2 * input_dims4)  # extract dim1 from flattened dim2
    input_idx_remainder = idx2 % (input_dims2 * input_dims4)
    input_idx2 = input_idx_remainder // input_dims4
    input_idx3 = idx3
    input_idx4 = input_idx_remainder % input_dims4
    
    # Calculate flat input index
    input_flat_idx = (input_idx0 * input_dims1 + input_idx1) * (input_dims2 * input_dims3 * input_dims4) + \
                    (input_idx2 * input_dims3 + input_idx3) * input_dims4 + input_idx4
    
    # Load and store
    val = tl.load(input_ptr + input_flat_idx, mask=mask)
    tl.store(output_ptr + offsets, val, mask=mask)

@torch.fx.wrap
def fused_flatten_transpose(x):
    """
    Perform fused flatten(2) + transpose(1, 2) operation
    """
    # Get input shape: assuming [dim0, dim1, dim2, dim3, dim4]
    input_shape = x.shape
    dim0, dim1, dim2, dim3, dim4 = input_shape
    
    # Operations:
    # 1. flatten(2): [dim0, dim1, dim2*dim3*dim4]
    # 2. transpose(1, 2): [dim0, dim2*dim3*dim4, dim1] -> [dim0, dim3, dim1*dim2*dim4]
    #    Actually, let me trace this more carefully
    
    # flatten(2) on [dim0, dim1, dim2, dim3, dim4]:
    # keeps dimensions 0 and 1, flattens dimensions 2, 3, 4
    # becomes: [dim0, dim1, dim2*dim3*dim4]
    
    # transpose(1, 2) on [dim0, dim1, dim2*dim3*dim4]:
    # swaps dimensions 1 and 2
    # becomes: [dim0, dim2*dim3*dim4, dim1]
    
    # But that's not right either. Let me look at the shapes more carefully.
    # From the input shapes, we have:
    # conv3d output should be [1, 768, 5, 14, 14]
    # flatten(2): keep first 2 dims, flatten from dim 2: [1, 768, 5*14*14] = [1, 768, 980]
    # transpose(1, 2): swap dims 1 and 2: [1, 980, 768]
    
    output_shape = [dim0, dim2 * dim3 * dim4, dim1]
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    n_elements = out.numel()
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_flatten_transpose_kernel[(num_programs,)](
        x,
        out,
        dim0, dim1, dim2, dim3, dim4,
        output_shape[0], output_shape[1], output_shape[2],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_flatten_transpose