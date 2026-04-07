import torch
import triton
import triton.language as tl

# Pattern matching function - exactly matches the computation in model.py
def linear_reshape_softmax_pattern(in_0, in_1, in_2):
    """
    Pattern matching for linear + reshape + softmax computation sequence.
    Must exactly mirror the operations in model.py.
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    tmp_4 = torch.softmax(tmp_3, dim=1)
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for the replacement function"""
    return (in_0, in_1, in_2)

# Optimized Triton kernel that fuses linear, reshape, and softmax
@triton.jit
def fused_linear_reshape_softmax_kernel(
    bias_ptr,           # Input bias [18]
    weight_ptr,         # Input weight [18, 128]  
    input_ptr,          # Input [1, 19, 128]
    output_ptr,         # Output [342, 9, 1]
    batch_size: tl.constexpr,      # 1
    seq_len: tl.constexpr,        # 19
    hidden_dim: tl.constexpr,     # 128 
    num_heads: tl.constexpr,      # 9
    head_dim_2: tl.constexpr,     # 2 (since 18/9=2)
    bias_stride: tl.constexpr,    # 18
    weight_hidden_stride: tl.constexpr,  # 128
    weight_head_stride: tl.constexpr,    # 18*128=2304
    input_seq_stride: tl.constexpr,      # 19*128=2432
    input_hidden_stride: tl.constexpr,  # 128
    output_head_stride: tl.constexpr,    # 342
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of output elements
    pid = tl.program_id(0)
    
    # Calculate output coordinates: [342, 9, 1]
    output_idx = pid  # Linear index from 0 to 342*9-1
    
    # Convert linear index to 3D coordinates [342, 9, 1]
    head_idx = output_idx // 342
    element_idx = (output_idx % 342) // num_heads
    
    # Calculate corresponding input and weight indices
    seq_idx = element_idx // head_dim_2
    local_head_idx = element_idx % head_dim_2
    
    # Bias index (same for all heads)
    bias_idx = head_idx
    
    # Load bias [18]
    bias_val = tl.load(bias_ptr + bias_idx)
    
    # Compute weighted sum for this position
    weighted_sum = bias_val
    
    # Loop over hidden dimension (128) to compute dot product
    for h_offset in range(0, hidden_dim, BLOCK_SIZE):
        # Compute offsets within current block
        h_end = min(h_offset + BLOCK_SIZE, hidden_dim)
        h_mask = h_offset + tl.arange(0, BLOCK_SIZE) < h_end
        
        # Load weight chunk: [18, 128] -> [BLOCK_SIZE]
        weight_offset = weight_head_stride * head_idx + h_offset
        weight_chunk = tl.load(weight_ptr + weight_offset + tl.arange(0, BLOCK_SIZE),
                              mask=h_mask, other=0.0)
        
        # Load input chunk: [1, 19, 128] -> [BLOCK_SIZE]  
        input_offset = input_seq_stride * seq_idx + h_offset
        input_chunk = tl.load(input_ptr + input_offset + tl.arange(0, BLOCK_SIZE),
                             mask=h_mask, other=0.0)
        
        # Accumulate dot product
        weighted_sum += tl.sum(weight_chunk * input_chunk)
    
    # Compute exponential for softmax
    exp_val = tl.exp(weighted_sum - tl.max(weighted_sum))  # Numerical stability
    
    # Store result (we're computing this for each head separately)
    output_offset = output_idx
    tl.store(output_ptr + output_offset, exp_val)

@torch.fx.wrap
def fused_linear_reshape_softmax_computation(in_0, in_1, in_2):
    """
    Optimized wrapper function that launches the fused kernel
    """
    # Input shapes
    bias_shape = [18]
    weight_shape = [18, 128]
    input_shape = [1, 19, 128]
    
    # Output shape after reshape: [1, 19, 18] -> [342, 9, 1]
    total_output_elements = 19 * 18  # 342
    num_heads = 9
    
    # Create output tensor
    output_shape = [total_output_elements, num_heads, 1]
    output = torch.empty(output_shape, dtype=in_2.dtype, device=in_2.device)
    
    # Kernel launch parameters
    BLOCK_SIZE = 128  # Match hidden dimension for optimal memory coalescing
    num_programs = total_output_elements * num_heads
    
    # Define strides
    bias_stride = bias_shape[0]
    weight_hidden_stride = weight_shape[1]
    weight_head_stride = weight_shape[0] * weight_shape[1]
    input_seq_stride = input_shape[1] * input_shape[2]
    input_hidden_stride = input_shape[2]
    output_head_stride = 1  # Last dimension is 1
    
    # Launch fused kernel
    fused_linear_reshape_softmax_kernel[(num_programs,)](
        bias_ptr=in_0,
        weight_ptr=in_1,
        input_ptr=in_2,
        output_ptr=output,
        batch_size=1,
        seq_len=19,
        hidden_dim=128,
        num_heads=9,
        head_dim_2=2,  # 18/9=2
        bias_stride=bias_stride,
        weight_hidden_stride=weight_hidden_stride,
        weight_head_stride=weight_head_stride,
        input_seq_stride=input_seq_stride,
        input_hidden_stride=input_hidden_stride,
        output_head_stride=output_head_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function - returns reference to optimized kernel
def replacement_func():
    return fused_linear_reshape_softmax_computation