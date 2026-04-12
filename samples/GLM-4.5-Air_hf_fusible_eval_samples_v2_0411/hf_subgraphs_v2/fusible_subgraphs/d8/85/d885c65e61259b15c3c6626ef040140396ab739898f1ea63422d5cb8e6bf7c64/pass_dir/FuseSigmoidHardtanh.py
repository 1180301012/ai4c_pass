import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(conv2d_out, gate_input):
    """
    Match the pattern: sigmoid(conv2d_out) followed by hardtanh(conv2d_out * gate_input)
    This pattern appears in all the test cases with different data types
    """
    tmp_3 = conv2d_out.sigmoid()
    tmp_4 = gate_input * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5

# Argument extraction function  
def replacement_args(conv2d_out, gate_input):
    """
    Extract arguments needed for the replacement kernel
    """
    return (conv2d_out, gate_input)

# Optimized kernel that fuses sigmoid and hardtanh operations
@triton.jit
def fused_sigmoid_hardtanh_kernel(
    input_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    High-performance kernel that fuses:
    - element-wise multiplication with gate
    - sigmoid activation  
    - hardtanh clipping
    """
    # Each program handles a contiguous block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and gate tensors with proper casting for mixed precision
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    gate_val = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    
    # First apply gate multiplication (prevents sigmoid from being too extreme)
    gated_result = input_val * gate_val
    
    # Sigmoid with better numerical stability for all dtypes
    # Use simple sigmoid: 1 / (1 + exp(-input))
    exp_neg_input = tl.exp(-tl.abs(gated_result))
    if input_val.dtype in [tl.float16, tl.bfloat16]:
        # For lower precision, use a simpler approach
        sigmoid_result = tl.where(gated_result > 0,
                                 1.0 / (1.0 + exp_neg_input),
                                 exp_neg_input / (1.0 + exp_neg_input))
    else:
        # For fp32, use the standard sigmoid
        sigmoid_result = 1.0 / (1.0 + tl.exp(-gated_result))
    
    # Hardtanh: clamp between 0 and 6
    # This is simply: out = max(0, min(x, 6))
    result = sigmoid_result  # No need for hardtanh since sigmoid is already [0,1] and gate scales appropriately
    
    # Apply hardtanh only if needed (clamp to [0,6] range)
    result = tl.where(result < 0, 0.0, 
                     tl.where(result > 6.0, 6.0, result))
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

# Kernel wrapper for different data types
@torch.fx.wrap
def fused_sigmoid_hardtanh(conv2d_out, gate_input):
    """
    Wrapper that handles different data types and launches the kernel
    """
    # The fused operation should produce output with same shape as gate_input (in_2)
    # since that's what gets returned in the model
    output_shape = gate_input.shape
    n_elements = gate_input.numel()
    dtype = gate_input.dtype
    
    # Create output tensor with same properties as gate_input
    out = torch.empty_like(gate_input)
    
    # Block size configuration for GPU efficiency
    BLOCK_SIZE = 1024 if n_elements >= 1024 else 256
    
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with appropriate data type handling
    if dtype == torch.float16:
        # Specialized for float16
        fused_sigmoid_hardtanh_kernel[(num_programs,)](
            conv2d_out,
            gate_input,
            out,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif dtype == torch.bfloat16:
        # Specialized for bfloat16
        fused_sigmoid_hardtanh_kernel[(num_programs,)](
            conv2d_out,
            gate_input,
            out,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Fallback for float32
        fused_sigmoid_hardtanh_kernel[(num_programs,)](
            conv2d_out,
            gate_input,
            out,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

# Replacement function
def replacement_func():
    """
    Return the optimized kernel function
    """
    return fused_sigmoid_hardtanh