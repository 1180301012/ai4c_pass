import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, p):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = linear * (1.0 - p)
    tmp_4 = tmp_3.transpose(1, 2)
    return (tmp_3, tmp_4)

def replacement_args(in_0, in_1, in_2, p):
    return (in_0, in_1, in_2, p)

@triton.jit
def optimized_kernel(
    input_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    bias_ptr: tl.tensor,
    output_ptr: tl.tensor,
    batch_size: tl.int32,
    seq_len: tl.int32,
    out_channels: tl.int32,
    feature_dim: tl.int32,
    p: tl.float32,
    BLOCK_SIZE: tl.constexpr
):
    # Each thread handles one (batch, out_channel) element
    batch_id = tl.program_id(0)
    out_channel = tl.program_id(1)
    
    # Initialize accumulator
    accum = tl.zeros([1], tl.float32)
    
    # Loop over feature dimension
    for feature in range(BLOCK_SIZE):
        # Compute input index: [batch, seq, feature]
        input_idx = batch_id * seq_len + feature
        # Compute weight index: [out_channel, feature]
        weight_idx = out_channel * feature_dim + feature
        
        input_val = tl.load(input_ptr + input_idx)
        weight_val = tl.load(weight_ptr + weight_idx)
        
        accum += input_val * weight_val
    
    # Load bias
    bias_val = tl.load(bias_ptr + out_channel)
    
    # Apply scaling and bias
    accum = accum + bias_val * (1.0 - p)
    
    # Store in output
    tl.store(output_ptr + (batch_id * out_channels + out_channel), accum)

@torch.fx.wrap
def optimized_wrapper(in_0, in_1, in_2, p):
    B, S, F = in_2.shape
    O = in_1.shape[0]
    
    # Create output tensors
    out_s = torch.empty((B, S, O), in_2.dtype)
    out_t = torch.empty((B, O, S), in_2.dtype)
    
    # Configure kernel grid
    grid = ((B + 1024 - 1) // 1024, O)
    
    optimized_kernel[grid](
        input_ptr=in_2,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=out_t,
        batch_size=B,
        seq_len=S,
        out_channels=O,
        feature_dim=F,
        p=p,
        BLOCK_SIZE=128
    )
    
    return (out_s, out_t)

def replacement_func():
    return optimized_wrapper