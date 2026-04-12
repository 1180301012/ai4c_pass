import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching for linear + sigmoid + view(32, 64, 1, 1) + multiplication fusion
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    tmp_4 = tmp_3.view(32, 64, 1, 1)
    tmp_5 = in_3 * tmp_4
    return (tmp_5,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_kernel_batch32(
    bias_ptr,
    weight_ptr,
    input_ptr,
    output_ptr,
    n_batch: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
):
    """
    Fused kernel for batch size 32 case
    """
    # Map program to output elements using 3D grid
    batch_id = tl.program_id(0)
    feature_id = tl.program_id(1)
    offset = tl.program_id(2)
    
    # Calculate spatial indices
    h_id = offset // width
    w_id = offset % width
    
    # Calculate output offset
    output_offset = (batch_id * out_features * height * width + 
                     feature_id * height * width + 
                     h_id * width + w_id)
    
    if batch_id >= n_batch or feature_id >= out_features or h_id >= height or w_id >= width:
        return
    
    # Load bias for this output feature (convert to float for computation)
    bias = tl.load(bias_ptr + feature_id).to(tl.float32)
    
    # Compute linear result for this (batch, feature) combination
    linear_result = bias
    for k in range(in_features):
        weight_val = tl.load(weight_ptr + feature_id * in_features + k).to(tl.float32)
        input_val = tl.load(input_ptr + batch_id * in_features + k).to(tl.float32)
        linear_result += weight_val * input_val
    
    # Apply sigmoid with float precision
    sigmoid_result = 1.0 / (1.0 + tl.exp(-linear_result))
    
    # Load input_3 value and apply multiplication
    input_3_val = tl.load(output_ptr + output_offset)
    # Convert sigmoid result back to original data type and multiply
    final_result = input_3_val * sigmoid_result.to(input_3_val.type.scalar)
    
    # Store result
    tl.store(output_ptr + output_offset, final_result)

@torch.fx.wrap
def fused_kernel_wrapper_batch32(in_0, in_1, in_2, in_3):
    """
    Wrapper function for batch size 32
    """
    n_batch, in_features = in_2.shape
    out_features = in_1.shape[0]
    _, _, height, width = in_3.shape
    
    # Create output tensor and use input_3 as base
    output = torch.as_tensor(in_3)
    
    # Launch kernel with 3D grid
    grid = (n_batch, out_features, height * width)
    fused_kernel_batch32[grid](
        bias_ptr=in_0,
        weight_ptr=in_1,
        input_ptr=in_2,
        output_ptr=output,
        n_batch=n_batch,
        in_features=in_features,
        out_features=out_features,
        height=height,
        width=width,
    )
    
    return output

def replacement_func():
    return fused_kernel_wrapper_batch32