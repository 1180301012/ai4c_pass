import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Pattern: linear followed by transpose of last two dimensions
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
    tmp_3 = tmp_2.transpose(-1, -2)
    return tmp_3

def replacement_args(in_0, in_1, in_2):
    # Extract: bias, weight, input
    return (in_0, in_1, in_2)

@triton.jit
def fused_linear_transpose_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr, 
    output_ptr,
    batch_size,
    seq_len,
    in_features,
    out_features,
    BLOCK_SIZE_I: tl.constexpr,
):
    # Each program handles a block of output elements
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Each program processes BLOCK_SIZE_I input features
    i_start = pid_m * BLOCK_SIZE_I
    i_end = min(i_start + BLOCK_SIZE_I, in_features)
    
    # Compute output element this block is responsible for
    batch = pid_n // out_features
    out_feat = pid_n % out_features
    seq_pos = 0  # Process first sequence position (can be extended to multiple positions)
    
    # Initialize accumulator
    accumulator = 0.0
    
    # Vectorized computation over input features
    for i in range(i_start, i_end):
        weight_offset = out_feat * in_features + i
        
        # Load weight vector
        weight_val = tl.load(weight_ptr + weight_offset, mask=True, other=0.0).to(tl.float32)
        
        # Load input vector for this batch and sequence position
        if batch < batch_size and seq_pos < seq_len:
            input_offset = batch * seq_len * in_features + seq_pos * in_features + i
            input_val = tl.load(input_ptr + input_offset, mask=True, other=0.0).to(tl.float32)
            accumulator += input_val * weight_val
    
    # Add bias
    bias_offset = bias_ptr + out_feat
    bias_val = tl.load(bias_offset, mask=True, other=0.0).to(tl.float32)
    
    # Final result for this block
    final_result = accumulator + bias_val
    
    # Store result
    if batch < batch_size and seq_pos < seq_len:
        output_offset = batch * out_features * seq_len + out_feat * seq_len + seq_pos
        tl.store(output_ptr + output_offset, final_result)

@torch.fx.wrap
def fused_linear_transpose_forward(bias, weight, input_tensor):
    batch_size, seq_len, in_features = input_tensor.shape
    out_features = weight.shape[0]
    
    # Output should be [batch_size, out_features, seq_len] due to transpose
    output_size = (batch_size, out_features, seq_len)
    
    # Use reasonable block size for better GPU utilization
    BLOCK_SIZE_I = 32   # Process 32 input features per work item
    
    # Calculate grid dimensions
    # Each work item handles one batch-output feature pair, processing a block of input features
    grid_m = (in_features + BLOCK_SIZE_I - 1) // BLOCK_SIZE_I
    grid_n = batch_size * out_features
    
    # Create output tensor
    output = torch.empty(output_size, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel with 2D grid
    fused_linear_transpose_kernel[(grid_m, grid_n)](
        bias_ptr=bias,
        weight_ptr=weight,
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        in_features=in_features,
        out_features=out_features,
        BLOCK_SIZE_I=BLOCK_SIZE_I
    )
    
    return output

def replacement_func():
    return fused_linear_transpose_forward