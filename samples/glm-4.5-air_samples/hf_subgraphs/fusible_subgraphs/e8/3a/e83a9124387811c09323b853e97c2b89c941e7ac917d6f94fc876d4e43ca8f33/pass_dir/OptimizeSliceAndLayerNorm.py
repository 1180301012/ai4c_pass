import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # Full pattern including slice: (((in_3 + in_2) * in_1) + in_0)[slice(None, None, None), 0]
    tmp_2 = in_3 + in_2
    tmp_3 = tmp_2 * in_1
    tmp_4 = tmp_3 + in_0
    tmp_6 = tmp_4[slice(None, None, None), 0]
    # Return both values like the original function
    return (tmp_4, tmp_6)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def optimized_slice_layernorm_kernel(
    bias_ptr,
    weight_ptr,
    tensor1_ptr,
    tensor2_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the hidden dimension
    hidden_idx = tl.program_id(0)
    
    # Calculate offset for the first sequence position (index 0 along dimension 1)
    offset = 0 * hidden_size + hidden_idx  # 0 is the sequence index we want
    
    # Load bias and weight (broadcast across batch)
    bias = tl.load(bias_ptr + hidden_idx, mask=hidden_idx < hidden_size, other=0.0)
    weight = tl.load(weight_ptr + hidden_idx, mask=hidden_idx < hidden_size, other=0.0)
    
    # Load tensor elements for the first sequence position
    tensor1_val = tl.load(tensor1_ptr + offset, mask=hidden_idx < hidden_size, other=0.0)
    tensor2_val = tl.load(tensor2_ptr + offset, mask=hidden_idx < hidden_size, other=0.0)
    
    # Fused computation: ((tensor2 + tensor1) * weight) + bias
    result = ((tensor2_val + tensor1_val) * weight) + bias
    
    # Store result directly - output is [batch_size, hidden_size]
    tl.store(out_ptr + batch_size * hidden_idx + hidden_idx, result, mask=hidden_idx < hidden_size)

@torch.fx.wrap
def optimized_slice_layernorm(in_0, in_1, in_2, in_3):
    # Determine tensor shapes
    hidden_size = in_0.shape[0]  # bias/weight shape is [hidden_size]
    batch_size = in_2.shape[0]   # batch dimension
    
    # For the full tensor, we need to compute all sequence positions
    if len(in_2.shape) == 3:
        seq_len = in_2.shape[1]
        out_full_shape = (batch_size, seq_len, hidden_size)
    else:
        seq_len = 1
        out_full_shape = (batch_size, hidden_size)
    
    # Create full tensor using the first kernel (modified to handle 3D tensors)
    out_full = torch.empty(out_full_shape, dtype=in_0.dtype, device=in_0.device)
    
    BLOCK_SIZE = 256  # Optimal block size for hidden dimension
    if len(in_2.shape) == 3:
        grid = (hidden_size, batch_size * seq_len)
    else:
        grid = (hidden_size, batch_size)
    
    optimized_slice_layernorm_kernel[grid](
        bias_ptr=in_0,
        weight_ptr=in_1,
        tensor1_ptr=in_2,
        tensor2_ptr=in_3,
        out_ptr=out_full,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Extract the slice
    if len(out_full.shape) == 3:
        out_sliced = out_full[:, 0, :]  # [batch_size, hidden_size]
    else:
        out_sliced = out_full
    
    return (out_full, out_sliced)

def replacement_func():
    return optimized_slice_layernorm