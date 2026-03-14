import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Match ReLU followed by dropout2d with training=False (inference mode)"""
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.dropout2d(tmp_0, 0.1, False, False)
    return (tmp_1, tmp_0)

def replacement_args(in_0):
    """Extract input tensor argument"""
    return (in_0,)

@triton.jit
def fused_relu_kernel(
    input_ptr,
    in_shape_ptr,
    out_ptr0,
    out_ptr1,
    num_elements: tl.constexpr,
):
    """Fused ReLU kernel for inference (dropout becomes identity)"""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_size = tl.num_programs(0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < num_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU activation
    relu_output = tl.maximum(input_data, 0.0)
    
    # Store results for both outputs (dropout is identity during inference)
    tl.store(out_ptr0 + offsets, relu_output, mask=mask)
    tl.store(out_ptr1 + offsets, relu_output, mask=mask)

@torch.fx.wrap
def fused_relu_dropout2d_gpu(in_0):
    """Fused ReLU + Dropout2d optimized for GPU inference"""
    # Get input tensor properties
    input_shape = in_0.shape
    input_ptr = in_0.data_ptr()
    
    # Calculate total number of elements
    total_elements = in_0.numel()
    
    # Create output tensors (same shape as ReLU output)
    out0 = torch.empty_like(in_0)
    out1 = torch.empty_like(in_0)
    
    # Set up Triton kernel execution
    block_size = 1024  # Optimized block size for GPU
    num_programs = (total_elements + block_size - 1) // block_size
    
    # Launch the fused kernel
    fused_relu_kernel[(num_programs,)](
        input_ptr=input_ptr,
        in_shape_ptr=input_shape,  # Passed but not used in kernel, keep for consistency
        out_ptr0=out0,
        out_ptr1=out1,
        num_elements=total_elements,
    )
    
    return out0, out1

def replacement_func():
    """Return the fused function"""
    return fused_relu_dropout2d_gpu