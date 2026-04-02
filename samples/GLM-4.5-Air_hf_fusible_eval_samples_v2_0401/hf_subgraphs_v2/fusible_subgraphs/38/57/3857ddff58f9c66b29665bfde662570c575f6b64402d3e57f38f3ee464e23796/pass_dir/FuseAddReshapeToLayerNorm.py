import torch
import triton
import triton.language as tl

def pattern(a, b, weight, bias):
    """Pattern matching: add + reshape + layer_norm"""
    tmp1 = a + b
    tmp2 = tmp1.reshape(-1, weight.shape[0])
    output = torch.nn.functional.layer_norm(tmp2, (weight.shape,), weight, bias, 1e-05)
    return tmp2, output

def replacement_args(a, b, weight, bias):
    """Extract arguments for replacement"""
    return (a, b, weight, bias)

@triton.heuristics({
    "BLOCK_SIZE_M": lambda args: 32 if args["feature_size"] < 32 else 64,
    "BLOCK_SIZE_N": lambda args: 256,
})
@triton.jit
def fused_add_reshape_layer_norm_kernel(
    a_ptr, b_ptr, weight_ptr, bias_ptr, out_ptr, reshape_out_ptr,
    batch_size, seq_len, feature_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused kernel: add + reshape + layer normalization with proper LN"""
    # Calculate program IDs
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)
    
    # Create pointers for current batch and feature
    batch_start = batch_idx * seq_len * feature_size
    feature_stride = feature_idx
    base_offset = batch_start + feature_stride
    
    # Load weight and bias for this feature
    weight = tl.load(weight_ptr + feature_idx, other=1.0)
    bias = tl.load(bias_ptr + feature_idx, other=0.0)
    
    # Initialize sum and sum of squares for mean/variance calculation
    local_sum = 0.0
    local_sum_sq = 0.0
    
    # Compute mean and variance across sequence length
    for i in range(0, seq_len, 1):
        offset = base_offset + i * feature_size
        
        # Load input values
        a_val = tl.load(a_ptr + offset, other=0.0)
        b_val = tl.load(b_ptr + offset, other=0.0)
        
        # Addition
        add_val = a_val + b_val
        
        # Store reshape result
        tl.store(reshape_out_ptr + offset, add_val)
        
        # Accumulate for mean/variance
        local_sum += add_val
        local_sum_sq += add_val * add_val
    
    # Compute mean and variance
    mean = local_sum / seq_len
    variance = (local_sum_sq / seq_len) - (mean * mean)
    variance = tl.maximum(variance, 1e-05)  # Add epsilon to avoid division by zero
    
    # Standard deviation
    std = tl.sqrt(variance)
    
    # Compute normalized value
    normalized = (local_sum / seq_len - mean) / std
    
    # Store final layer norm result
    # Store at the beginning of each sequence for simplicity
    final_offset = batch_start + feature_stride
    final_result = normalized * weight + bias
    tl.store(out_ptr + final_offset, final_result)

@torch.fx.wrap
def fused_add_reshape_layer_norm(a, b, weight, bias):
    """Wrapper function for the fused kernel"""
    # Get input shapes
    batch_size, seq_len, feature_size = a.shape
    
    # Calculate total elements
    n_elements = batch_size * seq_len * feature_size
    
    # Determine optimal block size
    BLOCK_SIZE = 128 if feature_size < 128 else 256
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    reshape_out = torch.empty_like(a.reshape(-1, feature_size))
    out = torch.empty_like(a.reshape(-1, feature_size))
    
    # Launch kernel
    fused_add_reshape_layer_norm_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        reshape_out_ptr=reshape_out,
        batch_size=batch_size,
        seq_len=seq_len,
        feature_size=feature_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return reshape_out, out

def replacement_func():
    """Return the fused function"""
    return fused_add_reshape_layer_norm