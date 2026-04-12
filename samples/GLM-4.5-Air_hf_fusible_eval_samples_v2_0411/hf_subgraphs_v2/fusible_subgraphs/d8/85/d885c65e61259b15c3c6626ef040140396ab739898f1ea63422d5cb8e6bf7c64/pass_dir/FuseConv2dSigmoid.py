import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(input_tensor, weight_tensor, bias_tensor, gate_input):
    """
    Match the pattern: conv2d -> sigmoid -> multiply with gate
    This includes the convolution operation that transforms features and produces attention weights
    """
    conv2d = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = gate_input * tmp_3
    return tmp_4

# Argument extraction function  
def replacement_args(input_tensor, weight_tensor, bias_tensor, gate_input):
    """
    Extract arguments needed for the replacement kernel
    """
    return (input_tensor, weight_tensor, bias_tensor, gate_input)

# Optimized kernel that fuses conv2d with sigmoid for channel transformation
@triton.jit
def fused_conv_sigmoid_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size, in_channels, out_channels,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    High-performance kernel that fuses:
    - 1x1 convolution for channel transformation
    - sigmoid activation to produce attention weights
    """
    # Program handles a block of output channels
    pid_m = tl.program_id(0)
    m_start = pid_m * BLOCK_SIZE_M
    m_offs = m_start + tl.arange(0, BLOCK_SIZE_M)
    
    # Mask for valid output channels
    m_mask = m_offs < out_channels
    
    # Load bias for this block of output channels
    bias_vals = tl.load(bias_ptr + m_offs, mask=m_mask, other=0.0)
    
    # Compute convolution and apply sigmoid
    for k in tl.range(0, in_channels, BLOCK_SIZE_K):
        k_offs = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offs < in_channels
        
        # Load input tensor (batch, in_channels, 1, 1) -> (batch, 1, in_channels)
        input_vals = tl.load(input_ptr + k_offs, mask=k_mask, other=0.0)
        
        # Load weights for this output channel block (out_channels, in_channels)
        weight_vals = tl.load(weight_ptr + m_offs[:, None] * in_channels + k_offs[None, :], 
                            mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Compute: conv = input @ weight^T + bias
        # Since this is 1x1 conv with 1x1 spatial, it's just matrix multiplication
        conv_result = tl.dot(weight_vals, input_vals) + bias_vals
        
        # Apply sigmoid: 1 / (1 + exp(-conv_result))
        sigmoid_result = 1.0 / (1.0 + tl.exp(-conv_result))
    
    # Store the sigmoid results (attention weights)
    tl.store(out_ptr + m_offs, sigmoid_result, mask=m_mask)

# Kernel wrapper for conv2d -> sigmoid fusion
@torch.fx.wrap
def fused_conv_sigmoid(input_tensor, weight_tensor, bias_tensor):
    """
    Wrapper that handles different data types and launches the kernel for conv2d + sigmoid
    """
    # Get tensor shapes
    batch_size, in_channels, out_channels = input_tensor.shape[0], weight_tensor.shape[1], weight_tensor.shape[0]
    
    # Conv2D with 1x1 kernel produces [batch_size, out_channels, 1, 1]
    # Reshape to [batch_size * out_channels] for processing
    n_elements = batch_size * out_channels
    
    # Create output tensor for sigmoid results
    out = torch.empty((batch_size, out_channels, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Block size configuration for GPU efficiency
    BLOCK_SIZE_M = 128  # Output channels per block (reduced for better occupancy)
    BLOCK_SIZE_K = 32   # Input channels per block (smaller for better shared memory usage)
    
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Launch kernel with appropriate data type handling
    dtype = input_tensor.dtype
    if dtype == torch.float16:
        fused_conv_sigmoid_kernel[(num_programs,)](
            input_tensor,
            weight_tensor,
            bias_tensor,
            out,
            batch_size, in_channels, out_channels,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
    elif dtype == torch.bfloat16:
        fused_conv_sigmoid_kernel[(num_programs,)](
            input_tensor,
            weight_tensor,
            bias_tensor,
            out,
            batch_size, in_channels, out_channels,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
    else:
        # Fallback for float32
        fused_conv_sigmoid_kernel[(num_programs,)](
            input_tensor,
            weight_tensor,
            bias_tensor,
            out,
            batch_size, in_channels, out_channels,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
    
    return out

# Helper function for complete pattern (conv2d + sigmoid + gate mult)
def complete_fused_operation(conv2d_out, gate_input):
    """
    Complete fused operation: sigmoid(conv2d_out) * gate_input
    This matches the full pattern in the computation graph
    """
    # Apply our optimized sigmoid computation
    sigmoid_result = fused_conv_sigmoid(conv2d_out)  # Actually conv2d already applied sigmoid
    
    # Then apply gate multiplication (this could also be optimized separately)
    gated_result = gate_input * sigmoid_result
    
    return gated_result

# Replacement function
def replacement_func():
    """
    Return the optimized kernel function for the complete fused operation
    """
    return complete_fused_operation