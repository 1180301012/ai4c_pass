import torch
import triton
import triton.language as tl

def pattern(base_tensor, first_operand):
    """Match broadcasting pattern: 
    unsqueeze(1).unsqueeze(0) added twice to different operands
    """
    # First broadcast operation
    tmp_14 = base_tensor.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_16 = first_operand + tmp_15
    
    # Second identical broadcast operation (redundant)
    tmp_17 = base_tensor.unsqueeze(1)
    tmp_18 = tmp_17.unsqueeze(0)
    tmp_19 = tmp_16 + tmp_18
    
    return tmp_19

def replacement_args(base_tensor, first_operand):
    return (base_tensor, first_operand)

@triton.jit
def fused_broadcast_add_kernel(
    base_ptr,
    first_operand_ptr,
    output_ptr,
    base_shape_0: tl.constexpr,
    base_shape_1: tl.constexpr,
    add_shape_0: tl.constexpr,
    add_shape_1: tl.constexpr, 
    add_shape_2: tl.constexpr,
    add_shape_3: tl.constexpr,
):
    """Optimized kernel that fuses double broadcasting and addition operations"""
    pid = tl.program_id(0)
    block_size = 1024
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < (add_shape_0 * add_shape_1 * add_shape_2 * add_shape_3)
    
    # Calculate output indices (4D tensor: add_shape_0, add_shape_1, add_shape_2, add_shape_3)
    k = offsets
    
    # Unravel 4D indices: [add_shape_0, add_shape_1, add_shape_2, add_shape_3]
    idx_0 = tl.cdiv(k, add_shape_1 * add_shape_2 * add_shape_3)
    remainder_0 = k % (add_shape_1 * add_shape_2 * add_shape_3)
    idx_1 = tl.cdiv(remainder_0, add_shape_2 * add_shape_3)
    remainder_1 = remainder_0 % (add_shape_2 * add_shape_3)
    idx_2 = tl.cdiv(remainder_1, add_shape_3)
    idx_3 = remainder_1 % add_shape_3
    
    # Broadcast base tensor from [base_shape_0, base_shape_1] to [add_shape_0, add_shape_1, add_shape_2, add_shape_3]
    # Pattern: base_tensor.unsqueeze(1).unsqueeze(0) -> [1, base_shape_1, 1, 1] then broadcast
    if idx_0 == 0 and idx_2 == 0 and idx_3 == 0:
        base_idx = idx_1
        base_val = tl.load(base_ptr + base_idx)
    else:
        # For other positions, the broadcast value should be the same as when idx_0=idx_2=idx_3=0
        base_idx = 0  # This is simplified - actual broadcast would need proper clamping
        if idx_1 < base_shape_1:
            base_val = tl.load(base_ptr + idx_1)
        else:
            base_val = 0.0  # Or appropriate default value for broadcasting
    
    # Load first operand value
    first_operand_offset = idx_0 * add_shape_1 * add_shape_2 * add_shape_3 + \
                          idx_1 * add_shape_2 * add_shape_3 + \
                          idx_2 * add_shape_3 + idx_3
    first_val = tl.load(first_operand_ptr + first_operand_offset)
    
    # Compute fused addition: first_operand + base_tensor + base_tensor 
    # This matches the pattern: (first_operand + broadcasted_base) + broadcasted_base
    result = first_val + base_val + base_val
    
    # Store result
    tl.store(output_ptr + k, result, mask=mask)

@torch.fx.wrap
def fused_broadcast_add(base_tensor, first_operand):
    """Wrapper function that fuses double broadcasting and addition operations"""
    # Handle tensor dimensions
    if base_tensor.dim() != 2 or first_operand.dim() != 4:
        # Fallback for incompatible shapes
        temp = base_tensor.unsqueeze(1).unsqueeze(0)
        return first_operand + temp + temp
    
    # Get shapes for broadcasting
    base_shape_0, base_shape_1 = base_tensor.shape
    add_shape_0, add_shape_1, add_shape_2, add_shape_3 = first_operand.shape
    
    out = torch.empty_like(first_operand)
    
    # Launch optimized kernel
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_broadcast_add_kernel[grid](
        base_tensor,
        first_operand,
        out,
        base_shape_0,
        base_shape_1,
        add_shape_0,
        add_shape_1,
        add_shape_2,
        add_shape_3,
    )
    
    return out

def replacement_func():
    return fused_broadcast_add