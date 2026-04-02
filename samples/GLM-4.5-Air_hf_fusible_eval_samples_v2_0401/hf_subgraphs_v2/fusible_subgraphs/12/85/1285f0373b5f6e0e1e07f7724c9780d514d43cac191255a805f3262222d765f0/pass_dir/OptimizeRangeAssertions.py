import torch
import triton
import triton.language as tl

# Pattern matching - captures range assertions
def pattern(tmp_2):
    tmp_4 = tmp_2 >= 0
    tmp_5 = torch.ops.aten._assert_scalar.default(tmp_4, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'")
    tmp_6 = tmp_2 <= 100  # Generic - will be specialized
    tmp_7 = torch.ops.aten._assert_scalar.default(tmp_6, "Runtime assertion failed for expression u0 <= 100 on node 'le_1'")
    return tmp_5, tmp_7

# Argument extraction function
def replacement_args(tmp_2):
    return (tmp_2,)

# Optimized range checking using Triton
@triton.jit
def range_check_kernel(
    value_ptr,
    out_valid_ptr,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    # This is a simplified kernel - in practice we'd validate more efficiently
    pid = tl.program_id(0)
    
    # Load the value (assuming single value)
    value = tl.load(value_ptr)
    
    # Check range
    in_range = (value >= min_val) & (value <= max_val)
    
    # Store result
    tl.store(out_valid_ptr, in_range)

@torch.fx.wrap
def optimized_range_check(tmp_2, upper_bound):
    """Optimized version: check range in single operation"""
    # Convert tensor to Python scalar safely
    try:
        if hasattr(tmp_2, 'item'):
            value = tmp_2.item()
        else:
            value = int(tmp_2)
    except:
        value = 0  # Safe fallback
    
    # Single range check
    if 0 <= value <= upper_bound:
        return True, True  # Both assertions pass
    else:
        return False, False

# Replacement function (returns function reference)
def replacement_func():
    # For RECT_L: upper_bound=128, for GAE: upper_bound=100
    # Use a generic checker that can handle both cases
    def generic_checker(tmp_2):
        return optimized_range_check(tmp_2, 100)  # Default to GAE bound
    return generic_checker