import torch
import triton
import triton.language as tl


def pattern(bias, weight, input):
    """
    Pattern: linear(input, weight, bias) followed by transpose(-1, -2)
    
    The pattern matches:
    tmp_0 = in_0  # bias
    tmp_1 = in_1  # weight  
    tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
    tmp_3 = tmp_2.transpose(-1, -2)
    """
    tmp_0 = bias
    tmp_1 = weight
    tmp_2 = torch.nn.functional.linear(input, tmp_1, tmp_0)
    tmp_3 = tmp_2.transpose(-1, -2)
    return tmp_3


def replacement_args(bias, weight, input):
    """Extract the arguments needed for the replacement kernel."""
    return (bias, weight, input)


@triton.jit
def fused_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, hidden, features,
    stride_in_b, stride_in_h, stride_in_f,
    stride_w_o, stride_w_f,
    stride_out_b, stride_out_o, stride_out_h,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused linear + transpose kernel.
    
    Each program computes a tile of output elements.
    Grid: (batch * features * hidden // BLOCK_SIZE, )
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Each program computes BLOCK_SIZE output elements
    element_start = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    
    # Calculate indices for each element in this block
    total_elements = batch * features * hidden
    
    # Skip elements outside range
    if element_start >= total_elements:
        return
    
    # Global index for each thread
    global_idx = element_start + offs
    valid_mask = global_idx < total_elements
    
    # Compute batch, out_feat, hidden indices
    batch_idx = global_idx // (features * hidden)
    rem = global_idx % (features * hidden)
    out_feat = rem // hidden
    hidden_idx = rem % hidden
    
    # Compute output pointer offsets
    out_offs = (batch_idx * stride_out_b + 
                out_feat * stride_out_o + 
                hidden_idx * stride_out_h)
    
    # Initialize accumulator
    acc = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
    
    # Iterate over features dimension
    for f in range(0, features):
        # Load weight[out_feat, f] - need to handle per-element weight load
        # Actually, we need to load weight for each valid output element
        # This gets complex - let's simplify by doing element-wise compute
        
        # For each element in the block:
        # output[b,o,h] = sum_f(input[b,h,f] * weight[o,f]) + bias[o]
        
        # Load input[b, h, f] for each element
        for elem_idx in range(BLOCK_SIZE):
            # Skip invalid elements
            if (batch_idx[elem_idx] >= batch) or (out_feat[elem_idx] >= features) or (hidden_idx[elem_idx] >= hidden):
                continue
                
            # Compute input offset
            i_off = (batch_idx[elem_idx] * stride_in_b + 
                     hidden_idx[elem_idx] * stride_in_h + 
                     f * stride_in_f)
            
            # Compute weight offset  
            w_off = (out_feat[elem_idx] * stride_w_o + f * stride_w_f)
            
            # Load and multiply
            i_val = tl.load(input_ptr + i_off)
            w_val = tl.load(weight_ptr + w_off)
            
            # Accumulate
            acc[elem_idx] = acc[elem_idx] + i_val * w_val
    
    # Add bias
    for elem_idx in range(BLOCK_SIZE):
        if (batch_idx[elem_idx] < batch) and (out_feat[elem_idx] < features) and (hidden_idx[elem_idx] < hidden):
            b_val = tl.load(bias_ptr + out_feat[elem_idx])
            acc[elem_idx] = acc[elem_idx] + b_val
    
    # Store result
    for elem_idx in range(BLOCK_SIZE):
        if (batch_idx[elem_idx] < batch) and (out_feat[elem_idx] < features) and (hidden_idx[elem_idx] < hidden):
            tl.store(output_ptr + out_offs[elem_idx], acc[elem_idx])


@torch.fx.wrap
def fused_linear_transpose_wrapper(bias, weight, input):
    """Pure Triton implementation of linear + transpose."""
    batch, hidden, features = input.shape
    
    # Allocate output
    output = torch.empty((batch, features, hidden), device=input.device, dtype=input.dtype)
    
    # Grid: total number of output elements
    grid = (batch * features * hidden,)
    
    BLOCK_SIZE = 64
    
    fused_kernel[grid](
        input, weight, bias, output,
        batch, hidden, features,
        input.stride(0), input.stride(1), input.stride(2),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    """Return the replacement function."""
    return fused_linear_transpose_wrapper