import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching for linear + sigmoid + view + multiplication fusion
    Must match exactly the structure in model.py
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    # Use dynamic view to handle different batch sizes
    batch_size = in_3.shape[0]
    tmp_4 = tmp_3.view(batch_size, 64, 1, 1)
    tmp_5 = in_3 * tmp_4
    return (tmp_5,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    output_ptr,
    n_batch: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel: Linear + Sigmoid + Broadcast + Multiplication
    Computes: output = input_3 * sigmoid(input_2 @ weight^T + bias)
    """
    # Map program to elements in output tensor
    batch_id = tl.program_id(0)
    feature_id = tl.program_id(1)
    h_id = tl.program_id(2)
    w_id = tl.program_id(3)
    
    # Compute output ptr offset
    output_offset = (batch_id * out_features * height * width + 
                     feature_id * height * width + 
                     h_id * width + w_id)
    
    if batch_id >= n_batch or feature_id >= out_features or h_id >= height or w_id >= width:
        return
    
    # Load bias for this output feature
    bias = tl.load(bias_ptr + feature_id)
    
    # Compute linear result for this (batch, feature) combination
    linear_result = bias
    for k in range(in_features):
        weight_val = tl.load(weight_ptr + feature_id * in_features + k)
        input_val = tl.load(input_ptr + batch_id * in_features + k)
        linear_result += weight_val * input_val
    
    # Apply sigmoid
    sigmoid_result = 1.0 / (1.0 + tl.exp(-linear_result))
    
    # Load input_3 value and apply multiplication
    input_3_val = tl.load(output_ptr + output_offset)
    final_result = input_3_val * sigmoid_result
    
    # Store result
    tl.store(output_ptr + output_offset, final_result)

@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2, in_3):
    """
    Wrapper function to launch the fused kernel
    """
    n_batch, in_features = in_2.shape
    out_features = in_1.shape[0]
    _, _, height, width = in_3.shape
    
    # Create output tensor
    output = torch.empty_like(in_3)
    
    # Copy input_3 to output as base
    output.copy_(in_3)
    
    # Configure block sizes for better GPU utilization
    BLOCK_SIZE_M = 64  # Block size for out_features
    BLOCK_SIZE_N = 8   # Block size for in_features  
    BLOCK_SIZE_K = 32  # Block size for threads
    
    # Calculate grid dimensions
    grid = (
        (n_batch + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (out_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        height,
        width
    )
    
    # Launch kernel
    fused_kernel[grid](
        bias_ptr=in_0,
        weight_ptr=in_1,
        input_ptr=in_2,
        output_ptr=output,
        n_batch=n_batch,
        in_features=in_features,
        out_features=out_features,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    return fused_kernel_wrapper