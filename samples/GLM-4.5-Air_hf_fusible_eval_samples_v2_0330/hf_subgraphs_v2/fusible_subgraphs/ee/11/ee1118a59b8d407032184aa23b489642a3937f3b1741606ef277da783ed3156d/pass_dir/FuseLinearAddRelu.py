import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Pattern matching Linear + Add + ReLU fusion"""
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = in_2 + linear
    tmp_4 = tmp_3.relu_()
    return tmp_4

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments for the fused kernel"""
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_linear_add_relu_kernel(
    bias_ptr,
    weight_ptr,
    residual_ptr,
    input_ptr,
    output_ptr,
    batch_size,
    in_features,
    out_features,
):
    """Simple but working fused Linear + Add + ReLU kernel"""
    # Each program handles one output element
    pid = tl.program_id(0)
    
    # Check bounds for 1D grid
    if pid >= batch_size * out_features:
        return
    
    # Calculate 2D coordinates from 1D index
    pid_m = pid // out_features
    pid_n = pid % out_features
    
    # Load bias once (it doesn't depend on k)
    bias_ok = (pid_n < out_features)
    b = tl.load(bias_ptr + pid_n, mask=bias_ok).to(tl.float32)
    
    # Load residual once (it doesn't depend on k)
    residual_ok = (pid_m < batch_size) & (pid_n < out_features)
    r = tl.load(residual_ptr + pid_m * out_features + pid_n, mask=residual_ok).to(tl.float32)
    
    # Compute matrix multiplication single element dot product
    acc = 0.0
    for k in range(in_features):
        # Get addresses for input and weight
        input_addr = pid_m * in_features + k
        weight_addr = pid_n * in_features + k
        
        # Create masks for input and weight loads
        input_ok = (pid_m < batch_size)  # k is bounded by loop
        weight_ok = (pid_n < out_features)  # k is bounded by loop
        
        # Load values and cast to float32 for computation
        x = tl.load(input_ptr + input_addr, mask=input_ok).to(tl.float32)
        w = tl.load(weight_ptr + weight_addr, mask=weight_ok).to(tl.float32)
        
        # Accumulate dot product
        acc += x * w
    
    # Add bias, residual, and apply ReLU
    result = max(acc + b + r, 0.0)
    
    # Store result
    output_addr = pid_m * out_features + pid_n
    tl.store(output_ptr + output_addr, result)

@torch.fx.wrap
def fused_linear_add_relu(bias, weight, residual, input):
    """Wrapper for the fused Linear + Add + ReLU operation"""
    batch_size, in_features = input.shape
    out_features = weight.shape[0]
    
    # Create output tensor
    output = torch.empty((batch_size, out_features), dtype=input.dtype, device=input.device)
    
    # Calculate grid dimensions (1D grid) - one program per output element
    total_elements = batch_size * out_features
    num_programs = total_elements
    
    # Launch kernel with 1D grid
    fused_linear_add_relu_kernel[(num_programs,)](
        bias_ptr=bias,
        weight_ptr=weight,
        residual_ptr=residual,
        input_ptr=input,
        output_ptr=output,
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
    )
    
    return output

def replacement_func():
    """Return the fused kernel function"""
    return fused_linear_add_relu