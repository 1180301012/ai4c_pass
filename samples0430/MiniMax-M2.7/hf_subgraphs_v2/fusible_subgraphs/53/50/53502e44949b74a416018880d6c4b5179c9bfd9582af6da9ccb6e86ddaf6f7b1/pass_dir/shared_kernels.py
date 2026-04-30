import torch
import triton
import triton.language as tl

# Route constants for dispatch
ROUTE_MULT_ADD_LN = "route_mult_add_ln"
ROUTE_ARANGE_EXPAND_ADD = "route_arange_expand_add"


# ============================================================================
# Triton Kernel for Multiply + Add + LayerNorm Fusion
# ============================================================================
@triton.jit
def multiply_add_layernorm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: ((x * scale) + y) -> LayerNorm
    """
    row_pid = tl.program_id(0)
    block_start = row_pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    scaled_x = x * scale
    sum_val = scaled_x + y
    
    row_offset = (offsets // hidden_size) * hidden_size
    row_offsets = row_offset + tl.arange(0, hidden_size)
    row_mask = row_offsets < n_elements
    
    row_x = tl.load(x_ptr + row_offsets, mask=row_mask, other=0.0)
    row_y = tl.load(y_ptr + row_offsets, mask=row_mask, other=0.0)
    row_scaled_x = row_x * scale
    row_sum = row_scaled_x + row_y
    
    row_mean = tl.sum(row_sum, axis=0) / hidden_size
    row_var = tl.sum((row_sum - row_mean) * (row_sum - row_mean), axis=0) / hidden_size
    
    normalized = (sum_val - row_mean) * tl.rsqrt(row_var + eps)
    
    col_offsets = offsets % hidden_size
    w = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    b = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    
    output = normalized * w + b
    tl.store(output_ptr + offsets, output, mask=mask)


# ============================================================================
# Triton Kernel for arange(start, end)
# ============================================================================
@triton.jit
def arange_kernel(
    output_ptr,
    start_val: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    result = start_val + offsets
    tl.store(output_ptr + offsets, result, mask=mask)


# ============================================================================
# Shared dispatch wrapper function
# ============================================================================
@torch.fx.wrap
def fused_dispatch_wrapper(
    route_arg, 
    in_0, in_1, in_2, in_3, in_4,
    arange_start=0, arange_end=0
):
    """
    Dispatch wrapper that routes to the appropriate optimized kernel.
    Shared by all passes to work within replacement_func_limit=1.
    """
    if route_arg == ROUTE_MULT_ADD_LN:
        # Route: Multiply + Add + LayerNorm fusion
        # in_1: token embedding output (scaled by 16.0 in graph)
        # in_0: position embedding output
        # in_3: layer_norm weight
        # in_2: layer_norm bias
        
        # Create scaled embedding from in_1 (multiply by 16.0)
        scaled_embedding = in_1 * 16.0
        
        # y is the position embedding (in_0)
        y = in_0
        
        # Weight and bias
        weight = in_3
        bias = in_2
        
        n_elements = scaled_embedding.numel()
        hidden_size = scaled_embedding.shape[-1]
        BLOCK_SIZE = min(256, n_elements)
        
        eps = 1e-05
        scale = 1.0  # Already applied to scaled_embedding
        
        # Allocate output
        output = torch.empty_like(scaled_embedding)
        
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        multiply_add_layernorm_kernel[(num_programs,)](
            scaled_embedding, y, weight, bias, output,
            n_elements, hidden_size, eps, scale, BLOCK_SIZE
        )
        return output
        
    elif route_arg == ROUTE_ARANGE_EXPAND_ADD:
        # Route: arange(arange_start, arange_start + 1).expand(1, -1) + arange_end
        # Simplified to: arange(arange_end + arange_start, arange_end + arange_start + 1)
        start_val = arange_end + arange_start
        n_elements = 1
        
        output = torch.empty((1, 1), dtype=torch.int64, device='cuda')
        
        BLOCK_SIZE = 256
        arange_kernel[(1,)](
            output, start_val, n_elements, BLOCK_SIZE
        )
        return output
    
    else:
        raise ValueError(f"Unknown route: {route_arg}")