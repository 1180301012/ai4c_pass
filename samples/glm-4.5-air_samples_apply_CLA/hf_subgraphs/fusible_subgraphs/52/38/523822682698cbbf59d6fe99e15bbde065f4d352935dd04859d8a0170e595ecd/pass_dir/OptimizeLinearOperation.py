import torch
import triton
import triton.language as tl

# Pattern matching function - matches linear operation
def pattern(in_6, in_5, in_4):
    # Linear operation: in_6 @ in_5.T + in_4
    # This corresponds to torch.nn.functional.linear(in_6, in_5, in_4)
    return torch.nn.functional.linear(in_6, in_5, in_4)

# Argument extraction function
def replacement_args(in_6, in_5, in_4):
    return (in_6, in_5, in_4)

# Optimized linear kernel using Triton
@triton.jit
def linear_kernel(
    x_ptr,           # Pointer to input tensor [batch, in_features]
    w_ptr,           # Pointer to weight tensor [out_features, in_features] 
    b_ptr,           # Pointer to bias tensor [out_features]
    out_ptr,         # Pointer to output tensor [batch, out_features]
    batch,           # Batch size
    in_features,     # Input features dimension
    out_features,    # Output features dimension
):
    # Program identifiers  
    m = tl.program_id(0)  # Batch index
    n = tl.program_id(1)  # Output feature index
    
    # Bounds check
    if m < batch and n < out_features:
        result = 0.0
        
        # Compute dot product over input features
        for k in range(in_features):
            # Use mask to ensure we're within bounds, though k should always be valid
            x_elem = tl.load(x_ptr + m * in_features + k, mask=k < in_features, other=0.0)
            w_elem = tl.load(w_ptr + n * in_features + k, mask=k < in_features, other=0.0)
            result += x_elem * w_elem
        
        # Add bias
        bias_elem = tl.load(b_ptr + n, mask=True, other=0.0)
        result += bias_elem
        
        # Store result with bounds mask
        tl.store(out_ptr + m * out_features + n, result, mask=True)

@torch.fx.wrap
def triton_linear(in_6, in_5, in_4):
    # Get tensor shapes
    batch, in_features = in_6.shape
    out_features, _ = in_5.shape
    
    # Output tensor
    out = torch.empty((batch, out_features), dtype=in_6.dtype, device=in_6.device)
    
    # Launch kernel - one thread per output element
    linear_kernel[(batch, out_features)](
        x_ptr=in_6,
        w_ptr=in_5,
        b_ptr=in_4,
        out_ptr=out,
        batch=batch,
        in_features=in_features,
        out_features=out_features,
    )
    
    return out

# Replacement function
def replacement_func():
    return triton_linear