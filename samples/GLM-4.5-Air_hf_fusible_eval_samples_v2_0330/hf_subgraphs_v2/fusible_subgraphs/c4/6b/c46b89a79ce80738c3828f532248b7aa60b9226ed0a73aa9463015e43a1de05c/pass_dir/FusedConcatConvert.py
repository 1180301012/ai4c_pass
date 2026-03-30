import torch
import triton
import triton.language as tl
import math

def pattern(tensor0, tensor1, tensor2):
    tmp_6 = torch.cat([tensor0, tensor1, tensor2], dim=0)
    tmp_7 = tmp_6.to(dtype=torch.float16)
    return tmp_7

def replacement_args(tensor0, tensor1, tensor2):
    return (tensor0, tensor1, tensor2)

@triton.jit
def fused_concat_convert_kernel(
    ptr0,
    ptr1, 
    ptr2,
    output_ptr,
    shape0_0,
    shape0_1,
    shape0_2, 
    shape0_3,
    shape1_0,
    shape1_1,
    shape1_2,
    shape1_3,
    shape2_0,
    shape2_1,
    shape2_2,
    shape2_3,
    output_0,
    output_1,
    output_2,
    output_3,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= output_0 * output_1 * output_2 * output_3:
        return
    
    # Calculate indices
    idx0 = pid // (output_1 * output_2 * output_3)
    remainder = pid % (output_1 * output_2 * output_3)
    idx1 = remainder // (output_2 * output_3)
    remainder = remainder % (output_2 * output_3)
    idx2 = remainder // output_3
    idx3 = remainder % output_3
    
    # Determine which input tensor this element comes from
    input_ptr = None
    input_idx = None
    
    if idx0 < shape0_0:
        # From tensor0
        input_ptr = ptr0
        input_idx = ((idx0 * shape0_1 + idx1) * shape0_2 + idx2) * shape0_3 + idx3
    elif idx0 < shape0_0 + shape1_0:
        # From tensor1  
        input_ptr = ptr1
        local_idx0 = idx0 - shape0_0
        input_idx = ((local_idx0 * shape1_1 + idx1) * shape1_2 + idx2) * shape1_3 + idx3
    else:
        # From tensor2
        input_ptr = ptr2
        local_idx0 = idx0 - shape0_0 - shape1_0
        input_idx = ((local_idx0 * shape2_1 + idx1) * shape2_2 + idx2) * shape2_3 + idx3
    
    # Load data and convert to float16
    data = tl.load(input_ptr + input_idx, other=0.0)
    output_value = data.to(tl.float16)
    
    # Store in output
    tl.store(output_ptr + pid, output_value)

@torch.fx.wrap  
def fused_concat_convert(tensor0, tensor1, tensor2):
    # Get shapes
    shape0 = tensor0.shape
    shape1 = tensor1.shape  
    shape2 = tensor2.shape
    
    # Calculate output shape
    output_shape = (shape0[0] + shape1[0] + shape2[0], shape0[1], shape0[2], shape0[3])
    output_elements = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=torch.float16, device=tensor0.device)
    
    BLOCK_SIZE = 128
    
    grid = (output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_concat_convert_kernel[grid](
        tensor0,
        tensor1,
        tensor2,
        output,
        shape0[0], shape0[1], shape0[2], shape0[3],
        shape1[0], shape1[1], shape1[2], shape1[3],
        shape2[0], shape2[1], shape2[2], shape2[3],
        output_shape[0], output_shape[1], output_shape[2], output_shape[3],
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_concat_convert