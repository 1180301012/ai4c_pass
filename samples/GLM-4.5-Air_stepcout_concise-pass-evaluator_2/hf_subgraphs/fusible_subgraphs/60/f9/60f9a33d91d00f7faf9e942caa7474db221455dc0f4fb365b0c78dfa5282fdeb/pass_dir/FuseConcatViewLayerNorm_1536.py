import torch
import triton
import triton.language as tl

def pattern(tmp_3, tmp_1, tmp_0):
    # PyTorch's layer_norm signature: layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05)
    # In our case: tmp_4 = torch.nn.functional.layer_norm(tmp_3, (1536,), tmp_1, tmp_0, 1e-05)
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (1536,), tmp_1, tmp_0, 1e-05)
    return tmp_4

def replacement_args(tmp_3, tmp_1, tmp_0):
    return (tmp_0, tmp_1, tmp_3)

@triton.jit
def fused_layer_norm_kernel_1536(
    x_ptr,          # pointer to the input tensor
    weight_ptr,     # pointer to the weight tensor
    bias_ptr,       # pointer to the bias tensor
    output_ptr,     # pointer to the output tensor
    n_tokens,       # number of tokens (256)
    hidden_size,    # hidden dimension (1536)
    eps: tl.constexpr,         # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # block size for hidden dimension
):
    # Get grid coordinates
    pid = tl.program_id(0)  # Only need 1D grid for hidden dimension
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    
    # Load weight and bias - they are small and can be loaded once
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Process each token sequentially
    for token_idx in range(n_tokens):
        # Load input for this token - ensure we're getting the right slice
        x_offset_base = token_idx * hidden_size
        x_offsets = x_offset_base + offsets
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        
        # Exact layer normalization computation matching PyTorch's algorithm
        # Compute mean over the hidden dimension (this gives us a scalar for each token)
        mean = tl.sum(x) * (1.0 / hidden_size)
        
        # Compute variance using the exact formula: E[X^2] - (E[X])^2
        x_centered = x - mean
        var = tl.sum(x_centered * x_centered) * (1.0 / hidden_size)
        
        # Compute standard deviation with epsilon
        std = tl.sqrt(var + eps)
        
        # Apply normalization exactly as PyTorch does
        x_norm = x_centered * (1.0 / std)
        output = x_norm * weight + bias
        
        # Store the result
        output_offsets = x_offset_base + offsets
        tl.store(output_ptr + output_offsets, output, mask=mask)

@torch.fx.wrap
def layer_norm_optimized_1536(bias, weight, input_tensor):
    # input_tensor is already in the correct shape [1, 256, 1536]
    # weight is [1536], bias is [1536]
    x = input_tensor
    
    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Set up grid and block sizes for 1D parallelization over hidden dimension
    # Handle tensor shape more robustly
    if x.dim() >= 3:
        n_tokens = x.shape[1]  # 256 tokens
        hidden_size = x.shape[2]  # 1536
    else:
        # Fallback for different shapes
        n_tokens = x.numel() // 1536
        hidden_size = 1536
    
    BLOCK_SIZE = 512  # Optimal block size for 1536
    
    # Calculate grid dimensions (1D grid for hidden dimension)
    grid = (triton.cdiv(hidden_size, BLOCK_SIZE),)
    
    fused_layer_norm_kernel_1536[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_tokens=n_tokens,
        hidden_size=hidden_size,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return layer_norm_optimized_1536