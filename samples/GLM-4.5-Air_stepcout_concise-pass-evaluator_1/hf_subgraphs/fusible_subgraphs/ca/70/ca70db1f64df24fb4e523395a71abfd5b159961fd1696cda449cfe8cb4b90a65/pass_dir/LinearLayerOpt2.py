import torch
import triton
import triton.language as tl

# Pattern matching function for second linear layer with batch dimensions
def pattern(input_tensor, weight, bias):
    """
    Pattern: torch.nn.functional.linear(input_tensor, weight, bias) 
    This corresponds to: tmp_10 = torch.nn.functional.linear(tmp_9, tmp_3, tmp_2)
    - input_tensor: [300, 1, 256] (reshaped from tmp_9)
    - weight: [512, 256] (tmp_3/in_3) 
    - bias: [512] (tmp_2/in_2)
    - Output: [300, 1, 512] then sliced to [300, 1, 256] results
    """
    return torch.nn.functional.linear(input_tensor, weight, bias)

# Argument extraction function
def replacement_args(input_tensor, weight, bias):
    return (input_tensor, weight, bias)

# Triton kernel for optimized linear layer with batch support
@triton.jit
def linear_kernel_batched(
    input_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    batch_stride,
    inner_dim,
    output_dim,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_INNER: tl.constexpr,
):
    """
    Optimized linear layer for batched tensors [batch, inner_batch, K] -> [batch, inner_batch, N]
    """
    # Program identifiers
    batch_pid = tl.program_id(0)
    inner_pid = tl.program_id(1)
    
    # Calculate positions
    batch_idx = batch_pid
    inner_idx = inner_pid
    
    if batch_idx >= batch_size or inner_idx >= inner_dim:
        return
    
    # Base pointer for this element
    input_base = input_ptr + batch_idx * batch_stride + inner_idx * 256
    out_base = out_ptr + batch_idx * batch_stride * 512 + inner_idx * 512
    
    # Use elementwise approach for each output
    batch_idx_local = batch_idx
    feat_idx = inner_idx
    
    # For each output feature in the last dimension
    for feat_n in range(0, 512):
        acc = 0.0
        
        # Accumulate over K (256 input features)
        for k in range(0, 256):
            # Load input element
            x_val = tl.load(input_base + k, other=0.0)
            
            # Load weight element [512, 256]
            weight_val = tl.load(weight_ptr + feat_n * 256 + k, other=0.0)
            
            # Multiply and accumulate
            acc += x_val * weight_val
        
        # Load bias and store
        bias_val = tl.load(bias_ptr + feat_n, other=0.0)
        result = acc + bias_val
        tl.store(out_base + feat_n, result, mask=feat_n < 512)
    
    

@torch.fx.wrap
def optimized_linear_batched(input_tensor, weight, bias):
    batch_size, inner_batch, K = input_tensor.shape
    N = bias.shape[0]
    
    # For linear layer: input [..., K] @ weight.t() [K, N] + bias [N] = output [..., N]
    # So weight should have shape [N, K]
    assert weight.shape[0] == N, f"Weight first dim {weight.shape[0]} must match bias dim {N}"
    assert weight.shape[1] == K, f"Weight second dim {weight.shape[1]} must match input dim {K}"
    
    output_shape = (batch_size, inner_batch, N)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid size
    grid = (batch_size, inner_batch)
    
    # Launch kernel
    linear_kernel_batched[grid](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output,
        batch_size=batch_size,
        batch_stride=inner_batch * K,
        inner_dim=inner_batch,
        output_dim=N,
        BLOCK_SIZE_BATCH=1,
        BLOCK_SIZE_INNER=1
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_linear_batched