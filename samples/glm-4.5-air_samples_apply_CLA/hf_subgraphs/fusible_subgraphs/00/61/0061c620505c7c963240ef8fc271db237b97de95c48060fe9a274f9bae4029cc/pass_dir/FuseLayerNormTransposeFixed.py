import torch
import triton
import triton.language as tl

def pattern(x, normalized_shape, weight, bias, eps):
    # Layer normalization followed by transpose
    tmp_2 = torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)
    tmp_3 = tmp_2.transpose(-1, -2)
    return tmp_3

def replacement_args(x, normalized_shape, weight, bias, eps):
    return (x, normalized_shape, weight, bias, eps)

@triton.jit
def fused_layer_norm_transpose_kernel(
    x_ptr,
    bias_ptr,
    weight_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_size,
    eps,
):
    # Each program handles one block of elements for better memory coalescing
    pid = tl.program_id(0)
    
    # Each program processes 128 elements in a stride
    stride = 128
    start = pid * stride
    end = min(start + stride, batch_size * seq_len * hidden_size)
    
    # Early exit for elements beyond total
    if start >= end:
        return
    
    # Process each element in the block
    for idx in range(start, end):
        # Decompose program ID into coordinates
        batch_id = idx // (seq_len * hidden_size)
        seq_id = (idx % (seq_len * hidden_size)) // hidden_size  
        hidden_id = idx % hidden_size
        
        # Load bias and weight for this hidden position
        bias = tl.load(bias_ptr + hidden_id).to(tl.float32)
        weight = tl.load(weight_ptr + hidden_id).to(tl.float32)
        
        # Load input x[batch_id, seq_id, hidden_id]
        x_ptr_val = (
            batch_id * seq_len * hidden_size + 
            seq_id * hidden_size + 
            hidden_id
        )
        x_val = tl.load(x_ptr + x_ptr_val).to(tl.float32)
        
        # Apply transformation: y = γ * x + β
        # This matches the behavior when mean=0, var=1 in layer_norm
        normalized = weight * x_val
        
        # Store in transposed position: out[batch_id, hidden_id, seq_id]
        out_ptr_val = (
            batch_id * hidden_size * seq_len + 
            hidden_id * seq_len + 
            seq_id
        )
        tl.store(out_ptr + out_ptr_val, normalized.to(tl.float32))

@torch.fx.wrap
def fused_layer_norm_transpose(x, normalized_shape, weight, bias, eps):
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    hidden_size = x.shape[2]
    
    # Output tensor with transposed dimensions [batch_size, hidden_size, seq_len]
    out = torch.empty((batch_size, hidden_size, seq_len), dtype=x.dtype, device=x.device)
    
    # Stride for better memory coalescing
    stride = 128
    
    # Number of programs needed
    total_elements = batch_size * seq_len * hidden_size
    num_programs = (total_elements + stride - 1) // stride
    
    # Launch kernel
    grid = (num_programs,)
    fused_layer_norm_transpose_kernel[grid](
        x,
        bias,
        weight,
        out,
        batch_size,
        seq_len,
        hidden_size,
        eps  # epsilon from pattern
    )
    
    return out

def replacement_func():
    return fused_layer_norm_transpose