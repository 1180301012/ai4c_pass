import torch
import triton
import triton.language as tl

def pattern(linear_input, weight_tensor, bias_tensor):
    """
    Pattern matching: Linear + Permute
    This handles the core matrix transformation pipeline
    Input format: linear_input (batch, seq, in_features), weight_tensor (out_features, in_features), bias_tensor (out_features)
    Output format: permuted to (batch, out_features, seq)
    """
    # Linear transformation
    tmp_2 = torch.nn.functional.linear(linear_input, weight_tensor, bias_tensor)
    
    # Permute dimensions from (batch, seq, out_features) to (batch, out_features, seq)
    tmp_3 = tmp_2.permute(0, 2, 1)
    
    return tmp_3

def replacement_args(linear_input, weight_tensor, bias_tensor):
    """Extract arguments needed for the replacement function"""
    return (linear_input, weight_tensor, bias_tensor)

@triton.jit
def optimized_linear_permute_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, seq_len, in_features, out_features,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel for Linear + Permute fusion
    Simple and efficient implementation using element-wise parallelism
    """
    pid = tl.program_id(0)
    
    # Calculate which element this program handles
    # Total elements = batch_size * seq_len * out_features
    total_elements = batch_size * seq_len * out_features
    if pid >= total_elements:
        return
    
    # Decode the linear index into batch, seq, and output feature dimensions
    output_idx = pid % out_features
    remaining = pid // out_features
    seq_idx = remaining % seq_len
    batch_idx = remaining // seq_len
    
    # Input offset: (batch_idx, seq_idx, in_features) -> linear index
    input_offset = batch_idx * seq_len * in_features + seq_idx * in_features
    
    # Weight offset: (output_idx, in_features) -> linear index  
    weight_offset = output_idx * in_features
    
    # Output offset with permute: (batch_idx, output_idx, seq_idx)
    output_offset = batch_idx * out_features * seq_len + output_idx * seq_len + seq_idx
    
    # Compute linear transformation
    result = 0.0
    for k in range(in_features):
        # Load input value with bounds checking
        input_offset_idx = input_offset + k
        input_mask = input_offset_idx < (batch_size * seq_len * in_features)
        input_val = tl.load(input_ptr + input_offset_idx, mask=input_mask, other=0.0)
        
        # Load weight value with bounds checking
        weight_offset_idx = weight_offset + k
        weight_mask = weight_offset_idx < (out_features * in_features)
        weight_val = tl.load(weight_ptr + weight_offset_idx, mask=weight_mask, other=0.0)
        
        # Multiply and accumulate
        result += input_val * weight_val
    
    # Add bias with bounds checking
    bias_mask = output_idx < out_features
    bias_val = tl.load(bias_ptr + output_idx, mask=bias_mask, other=0.0)
    result += bias_val
    
    # Store result with bounds checking
    output_mask = output_offset < total_elements
    tl.store(output_ptr + output_offset, result, mask=output_mask)

@torch.fx.wrap
def optimized_linear_permute(linear_input, weight_tensor, bias_tensor):
    """
    Optimized Linear + Permute implementation using Triton
    """
    batch_size = linear_input.shape[0]
    seq_len = linear_input.shape[1]
    in_features = linear_input.shape[2]
    out_features = weight_tensor.shape[0]
    
    # Output shape after permute: (batch, out_features, seq)
    output_shape = (batch_size, out_features, seq_len)
    output = torch.empty(output_shape, dtype=torch.float32, device=linear_input.device)
    
    # Configure block size for good GPU occupancy
    BLOCK_SIZE = 256  # Elements per thread block for good parallelism
    
    # Calculate total number of elements to process
    total_elements = batch_size * seq_len * out_features
    
    # Calculate grid size (1D grid)
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the simplified kernel - wrap grid_size in a tuple
    optimized_linear_permute_kernel[(grid_size,)](
        input_ptr=linear_input,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        in_features=in_features,
        out_features=out_features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the optimized function implementation"""
    return optimized_linear_permute