import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor):
    """Pattern: Linear + Permute fusion for position projection"""
    result = torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)
    final_result = result.permute(0, 3, 1, 2)
    return final_result

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

@triton.jit
def fused_linear_permute_kernel(
    input_ptr,
    weight_ptr, 
    bias_ptr,
    output_ptr,
    n: tl.constexpr,
    h: tl.constexpr,
    w: tl.constexpr,
    c_in: tl.constexpr,
    c_out: tl.constexpr,
):
    """Fused linear + permute kernel for position projection - simplified"""
    # Each program handles one output element
    pid = tl.program_id(0)
    
    # Convert program ID to output coordinates (N,C_out,H,W)
    w_idx = pid % w
    h_idx = (pid // w) % h
    c_out_idx = (pid // (w * h)) % c_out
    n_idx = pid // (w * h * c_out)
    
    # Check bounds
    if n_idx >= n or c_out_idx >= c_out:
        return
    
    # Load bias for this output channel
    bias_val = tl.load(bias_ptr + c_out_idx)
    acc = bias_val
    
    # Compute dot product: sum(input[n_idx,h_idx,w_idx,k] * weight[c_out_idx,k]) for k in [0, c_in)
    for k in range(c_in):
        # Load input element
        input_offset = (n_idx * h * w * c_in + h_idx * w * c_in + w_idx * c_in + k)
        input_val = tl.load(input_ptr + input_offset)
        
        # Load weight
        weight_offset = (c_out_idx * c_in + k)
        weight_val = tl.load(weight_ptr + weight_offset)
        
        # Accumulate
        acc += input_val * weight_val
    
    # Store result at permuted position [n_idx, c_out_idx, h_idx, w_idx]
    output_offset = (n_idx * c_out * h * w + c_out_idx * h * w + h_idx * w + w_idx)
    tl.store(output_ptr + output_offset, acc)

@torch.fx.wrap
def fused_linear_permute_kernel_wrapper(input_tensor, weight_tensor, bias_tensor):
    """Wrapper for fused linear + permute operation"""
    # Input tensor shape: [n, h, w, c_in] = [1, 196, 196, 3]
    # Weight tensor shape: [c_out, c_in] = [16, 3]  
    # Bias tensor shape: [c_out] = [16]
    # Output tensor shape: [n, c_out, h, w] = [1, 16, 196, 196]
    
    n, h, w, c_in = input_tensor.shape
    c_out = weight_tensor.shape[0]
    
    # Create output tensor with permuted dimensions
    output = torch.empty((n, c_out, h, w), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate number of programs (each handles one output element)
    total_elements = n * h * w * c_out
    num_programs = total_elements
    
    # Launch kernel
    fused_linear_permute_kernel[(num_programs,)](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        output_ptr=output,
        n=n,
        h=h,
        w=w,
        c_in=c_in,
        c_out=c_out,
    )
    
    return output

def replacement_func():
    return fused_linear_permute_kernel_wrapper