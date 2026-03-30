import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern: Independent linear transformation and element-wise multiplication
    Used in mmpose models where operations are independent
    
    This matches the computation:
    linear = torch.nn.functional.linear(in_3, in_0, None)
    tmp_3 = in_2 * in_1
    returns (tmp_3, linear)
    
    Args:
        in_0: Weight matrix for linear [out_features, in_features]
        in_1: Scaling factor for element-wise multiplication [features]
        in_2: Input tensor for element-wise multiplication [batch, seq, features]
        in_3: Input tensor for linear transformation [batch, seq, in_features]
    
    Returns:
        Tuple of (multiply_result, linear_result)
    """
    linear = torch.nn.functional.linear(in_3, in_0, None)
    tmp_3 = in_2 * in_1
    return tmp_3, linear

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments for the independent operations"""
    return (in_0, in_1, in_2, in_3)

@triton.jit
def efficient_linear_kernel(
    input_ptr,     # Input tensor [B, S, F]
    weight_ptr,    # Weight matrix [O, F]
    output_ptr,    # Output tensor [B, S, O]
    batch_size,    # Batch size
    seq_len,       # Sequence length
    in_features,   # Input features
    out_features,  # Output features
    # Strides
    input_stride_b,    # Batch stride for input
    input_stride_s,    # Seq stride for input  
    input_stride_f,    # Feature stride for input
    weight_stride_o,   # Output stride for weight
    weight_stride_f,   # Feature stride for weight
    output_stride_b,   # Batch stride for output
    output_stride_s,    # Seq stride for output
    output_stride_o,   # Output stride for output
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_F: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
):
    """Efficient linear transformation kernel"""
    
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_o = tl.program_id(2)
    
    # Compute ranges
    batch = pid_b
    seq = pid_s * BLOCK_SIZE_S
    feature_offset = pid_o * BLOCK_SIZE_O
    
    batch_end = min(batch + 1, batch_size)
    seq_end = min(seq + BLOCK_SIZE_S, seq_len)
    feature_end = min(feature_offset + BLOCK_SIZE_O, out_features)
    
    # Process output features
    for o_idx in range(feature_offset, feature_end):
        # Initialize accumulator
        acc = tl.zeros((seq_end - seq,), dtype=tl.float32)
        
        # Sum over input features
        for f_idx in range(in_features):
            # Load weight
            weight_val = tl.load(
                weight_ptr + o_idx * weight_stride_o + f_idx * weight_stride_f,
                mask=(o_idx < out_features),
                other=0.0
            ).to(tl.float32)
            
            # Load input for all sequential positions
            for s_idx in range(seq, seq_end):
                input_val = tl.load(
                    input_ptr + batch * input_stride_b + s_idx * input_stride_s + f_idx * input_stride_f,
                    mask=(batch < batch_size) and (s_idx < seq_end),
                    other=0.0
                ).to(tl.float32)
                
                acc += weight_val * input_val
        
        # Store results for all sequential positions
        for s_idx in range(seq, seq_end):
            tl.store(
                output_ptr + batch * output_stride_b + s_idx * output_stride_s + o_idx * output_stride_o,
                acc[s_idx - seq].to(tl.float16 if tl.load(weight_ptr).dtype == tl.float16 else tl.bfloat16),
                mask=(batch < batch_size) and (s_idx < seq_end) and (o_idx < out_features)
            )

@triton.jit
def efficient_elementwise_kernel(
    input_ptr,     # Input tensor [B, S, F]
    scale_ptr,     # Scale tensor [F] 
    output_ptr,    # Output tensor [B, S, F]
    batch_size,    # Batch size
    seq_len,       # Sequence length  
    features,      # Number of features
    # Strides
    input_stride_b,    # Batch stride for input
    input_stride_s,    # Seq stride for input
    input_stride_f,    # Feature stride for input
    scale_stride_f,    # Feature stride for scale
    output_stride_b,   # Batch stride for output
    output_stride_s,    # Seq stride for output
    output_stride_f,   # Feature stride for output
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_F: tl.constexpr,
):
    """Efficient element-wise multiplication with broadcasting"""
    
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_f = tl.program_id(2)
    
    # Compute ranges
    batch = pid_b
    seq = pid_s * BLOCK_SIZE_S
    feature = pid_f * BLOCK_SIZE_F
    
    batch_end = min(batch + 1, batch_size)
    seq_end = min(seq + BLOCK_SIZE_S, seq_len)
    feature_end = min(feature + BLOCK_SIZE_F, features)
    
    # Broadcast scale and process
    for f_idx in range(feature, feature_end):
        # Load scale value
        scale_val = tl.load(
            scale_ptr + f_idx * scale_stride_f,
            mask=(f_idx < features),
            other=0.0
        ).to(tl.float32)
        
        # Process all sequential positions for this feature
        for s_idx in range(seq, seq_end):
            # Load input value
            input_val = tl.load(
                input_ptr + batch * input_stride_b + s_idx * input_stride_s + f_idx * input_stride_f,
                mask=(batch < batch_size) and (s_idx < seq_end) and (f_idx < features),
                other=0.0
            ).to(tl.float32)
            
            # Store result
            result = input_val * scale_val
            tl.store(
                output_ptr + batch * output_stride_b + s_idx * output_stride_s + f_idx * output_stride_f,
                result.to(tl.float16 if tl.load(input_ptr).dtype == tl.float16 else tl.bfloat16),
                mask=(batch < batch_size) and (s_idx < seq_end) and (f_idx < features)
            )

@torch.fx.wrap
def optimized_linear_elementwise(in_0, in_1, in_2, in_3):
    """
    Optimized implementation of independent linear and element-wise operations
    Uses separate Triton kernels for each operation
    
    Args:
        in_0: Weight matrix [out_features, in_features]
        in_1: Scaling factor [features]  
        in_2: Input tensor for element-wise [batch, seq, features]
        in_3: Input tensor for linear [batch, seq, in_features]
    """
    # Get shapes
    if len(in_3.shape) == 3:
        batch_size, seq_len, in_features = in_3.shape
    else:
        batch_size = in_3.shape[0] // (in_3.shape[1] * in_3.shape[2])
        seq_len = in_3.shape[1]
        in_features = in_3.shape[2]
    
    out_features = in_0.shape[0]
    features = in_2.shape[-1]
    
    # Create output tensors
    linear_output = torch.empty((batch_size, seq_len, out_features), 
                               dtype=in_3.dtype, 
                               device=in_3.device)
    
    multiply_output = torch.empty_like(in_2)
    
    # Efficient linear transformation
    if len(in_3.shape) == 3:
        # Calculate strides
        input_stride_b = in_3.stride(0)
        input_stride_s = in_3.stride(1)
        input_stride_f = in_3.stride(2)
        
        weight_stride_o = in_0.stride(0)
        weight_stride_f = in_0.stride(1)
        
        output_stride_b = linear_output.stride(0)
        output_stride_s = linear_output.stride(1)
        output_stride_o = linear_output.stride(2)
        
        # Block sizes
        BLOCK_SIZE_S = 64
        BLOCK_SIZE_F = 32
        BLOCK_SIZE_O = 64
        
        # Grid dimensions
        grid_b = batch_size
        grid_s = (seq_len + BLOCK_SIZE_S - 1) // BLOCK_SIZE_S
        grid_o = (out_features + BLOCK_SIZE_O - 1) // BLOCK_SIZE_O
        
        # Launch linear kernel
        efficient_linear_kernel[(
            grid_b,
            grid_s, 
            grid_o
        )](
            input_ptr=in_3,
            weight_ptr=in_0,
            output_ptr=linear_output,
            batch_size=batch_size,
            seq_len=seq_len,
            in_features=in_features,
            out_features=out_features,
            input_stride_b=input_stride_b,
            input_stride_s=input_stride_s,
            input_stride_f=input_stride_f,
            weight_stride_o=weight_stride_o,
            weight_stride_f=weight_stride_f,
            output_stride_b=output_stride_b,
            output_stride_s=output_stride_s,
            output_stride_o=output_stride_o,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_F=BLOCK_SIZE_F,
            BLOCK_SIZE_O=BLOCK_SIZE_O,
        )
    
    # Efficient element-wise multiplication
    multiply_stride_b = in_2.stride(0)
    multiply_stride_s = in_2.stride(1)
    multiply_stride_f = in_2.stride(2)
    
    scale_stride_f = in_1.stride(0)
    
    output_stride_b = multiply_output.stride(0)
    output_stride_s = multiply_output.stride(1)
    output_stride_f = multiply_output.stride(2)
    
    # Block sizes
    BLOCK_SIZE_S = 64
    BLOCK_SIZE_F = 128
    
    # Grid dimensions
    grid_b = batch_size
    grid_s = (seq_len + BLOCK_SIZE_S - 1) // BLOCK_SIZE_S
    grid_f = (features + BLOCK_SIZE_F - 1) // BLOCK_SIZE_F
    
    # Launch element-wise kernel
    efficient_elementwise_kernel[(
        grid_b,
        grid_s,
        grid_f
    )](
        input_ptr=in_2,
        scale_ptr=in_1,
        output_ptr=multiply_output,
        batch_size=batch_size,
        seq_len=seq_len,
        features=features,
        input_stride_b=multiply_stride_b,
        input_stride_s=multiply_stride_s,
        input_stride_f=multiply_stride_f,
        scale_stride_f=scale_stride_f,
        output_stride_b=output_stride_b,
        output_stride_s=output_stride_s,
        output_stride_f=output_stride_f,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_F=BLOCK_SIZE_F,
    )
    
    return multiply_output, linear_output

def replacement_func():
    return optimized_linear_elementwise