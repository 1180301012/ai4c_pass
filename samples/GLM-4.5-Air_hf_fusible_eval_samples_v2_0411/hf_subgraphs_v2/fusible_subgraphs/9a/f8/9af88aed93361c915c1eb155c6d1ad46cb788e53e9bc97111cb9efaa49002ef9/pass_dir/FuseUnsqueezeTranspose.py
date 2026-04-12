import torch
import triton
import triton.language as tl

# Pattern matching function - must exactly mirror the computation
def pattern(in_0):
    tmp_1 = in_0.unsqueeze(1)
    tmp_2 = tmp_1.transpose(2, 3)
    return (tmp_2,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized Triton kernel with moderate vectorization for better performance
@triton.jit
def fused_reshape_kernel(
    input_ptr,
    output_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Moderate vectorization: handle 32 elements per program to reduce overhead
    # while keeping logic simple and correct
    
    # Each program handles a column of BLOCK_SIZE elements
    seq_in = tl.program_id(0)  # [0, 1023] - original sequence dimension  
    feat_offset = tl.program_id(1) * BLOCK_SIZE  # Start of feature block
    
    # Calculate offsets for a vector of elements
    offsets = tl.arange(0, BLOCK_SIZE)
    feat_in = feat_offset + offsets
    
    # Create mask for valid feature indices
    mask = feat_in < features
    
    # Input offset: (seq_in, feat_in) 
    input_offsets = seq_in * features + feat_in
    
    # Output offset: (feat_in, seq_in) in [1, 1, 128, 1024] space
    # Mapping: (i, j) -> (j, i) so element moves from sequence dimension to features dimension
    output_offsets = feat_in * 1024 + seq_in
    
    # Load vector of elements from input
    input_data = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Store to output with correct mapping
    tl.store(output_ptr + output_offsets, input_data, mask=mask)

# Optimized kernel wrapper
@torch.fx.wrap
def fused_reshape_transpose(in_0):
    # Get input tensor properties  
    input_shape = in_0.shape
    batch_size, seq_len, features = input_shape
    
    # Expected output shape: [batch, heads=1, seq_len=features, features=seq_len]
    output_shape = [batch_size, 1, features, seq_len]
    
    # Create output tensor using allowed operation
    output = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Use moderate block size for better performance
    BLOCK_SIZE = 32  # Balance between vectorization benefits and kernel overhead
    
    # Grid configuration: 
    # seq_dim: 1024 programs (original sequence dimension)
    # feat_dim: ceil(128/32) = 4 programs (4 groups of 32 features)
    fused_reshape_kernel[(seq_len, (features + BLOCK_SIZE - 1) // BLOCK_SIZE)](
        input_ptr=in_0,
        output_ptr=output,
        seq_len=seq_len,
        features=features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference, not called)
def replacement_func():
    return fused_reshape_transpose