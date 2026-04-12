import torch
import triton
import triton.language as tl

# Pattern matching function - match multiple consecutive element-wise operations
def pattern(conv2d_out, gate_input):
    """
    Match the pattern: sigmoid(conv2d_out) followed by multiplication and hardtanh
    This removes intermediate tensor allocations
    """
    tmp_3 = conv2d_out.sigmoid()
    tmp_4 = gate_input * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5

# Argument extraction function  
def replacement_args(conv2d_out, gate_input):
    """
    Extract arguments needed for the replacement kernel
    """
    return (conv2d_out, gate_input)

# Optimized kernel that fuses multiple element-wise operations
@triton.jit
def fused_elementwise_kernel(
    conv2d_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    High-performance kernel that fuses:
    - sigmoid activation
    - element-wise multiplication 
    - hardtanh clipping
    All in one kernel to eliminate intermediate allocations
    """
    # Each program handles a contiguous block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load tensors from their storage locations
    # Since tensors might be non-contiguous, access via flattening
    conv2d_flat = tl.load(conv2d_ptr + offsets, mask=mask, other=0.0)
    gate_flat = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation with minimal memory operations
    # Step 1: Apply sigmoid to conv2d result
    # Use numerically stable sigmoid for all dtypes
    exp_neg_conv = tl.exp(-tl.abs(conv2d_flat))
    sigmoid_result = tl.where(conv2d_flat > 0,
                             1.0 / (1.0 + exp_neg_conv),
                             exp_neg_conv / (1.0 + exp_neg_conv))
    
    # Step 2: Multiply with gate
    gated_result = sigmoid_result * gate_flat
    
    # Step 3: Apply hardtanh (clamp to [0,6])
    # This is more efficient than full hardtanh since sigmoid is already [0,1]
    # and gate multiplication typically results in reasonable values
    result = tl.where(gated_result < 0, 0.0,
                     tl.where(gated_result > 6.0, 6.0, gated_result))
    
    # Store final result directly
    tl.store(out_ptr + offsets, result, mask=mask)

# Optimized wrapper for small tensors (use vectorized ops)
@torch.fx.wrap  
def fast_elementwise_fusion(conv2d_out, gate_input):
    """
    Small tensor optimization: avoid Triton overhead for small operations
    """
    # For small tensors, just vectorized operations (no Triton kernel)
    result = conv2d_out.sigmoid()
    result = gate_input * result
    result = torch.clamp(result, 0.0, 6.0)
    return result

# Large tensor optimization with Triton
@torch.fx.wrap
def triton_elementwise_fusion(conv2d_out, gate_input):
    """
    Large tensor optimization: use Triton for better performance
    """
    # Determine total elements and use appropriate method
    conv_total_elements = conv2d_out.numel()
    gate_total_elements = gate_input.numel()
    
    # Determine output size (should be same as gate_input)
    out = torch.empty_like(gate_input)
    
    # Use simple approach for small tensors
    if conv_total_elements < 1024 and gate_total_elements < 1024:
        return fast_elementwise_fusion(conv2d_out, gate_input)
    
    # Use Triton for larger tensors with better block sizing
    BLOCK_SIZE = 256  # Smaller block size for better occupancy
    
    # For now, flatten tensors to 1D for simple processing
    # In production, would handle multi-dimensional tensors properly
    conv_flat = conv2d_out.flatten()
    gate_flat = gate_input.flatten()
    
    num_programs = (gate_total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    dtype = conv2d_out.dtype
    fused_elementwise_kernel[(num_programs,)](
        conv_flat, 
        gate_flat,
        out.flatten(),
        gate_total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Dispatch wrapper - chooses optimal method based on tensor size
@torch.fx.wrap
def optimized_fused_elementwise(conv2d_out, gate_input):
    """
    Dispatch to optimal implementation based on tensor characteristics
    """
    # Check if we can use the simple fast path
    conv_elements = conv2d_out.numel()
    gate_elements = gate_input.numel()
    
    # Use simple path for tensors with total elements < 2048
    if conv_elements < 1024 and gate_elements < 1024:
        return fast_elementwise_fusion(conv2d_out, gate_input)
    else:
        return triton_elementwise_fusion(conv2d_out, gate_input)

# Replacement function
def replacement_func():
    """
    Return the optimized kernel function
    """
    return optimized_fused_elementwise