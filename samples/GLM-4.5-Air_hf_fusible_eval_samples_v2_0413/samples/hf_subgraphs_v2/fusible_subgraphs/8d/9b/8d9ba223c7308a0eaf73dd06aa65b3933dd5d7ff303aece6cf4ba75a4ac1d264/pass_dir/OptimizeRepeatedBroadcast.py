import torch
import triton
import triton.language as tl

def pattern(base_tensor, first_operand):
    """Match repeated broadcasting pattern with unsqueeze operations"""
    # This matches the pattern:
    # tmp_14 = base_tensor.unsqueeze(1)
    # tmp_15 = tmp_14.unsqueeze(0)  
    # tmp_16 = first_operand + tmp_15
    # tmp_17 = base_tensor.unsqueeze(1)
    # tmp_18 = tmp_17.unsqueeze(0)
    # tmp_19 = tmp_16 + tmp_18
    
    # Instead of doing the broadcast twice, compute it once and reuse
    broadcasted = base_tensor.unsqueeze(1).unsqueeze(0)
    tmp_16 = first_operand + broadcasted
    tmp_19 = tmp_16 + broadcasted
    return tmp_19

def replacement_args(base_tensor, first_operand):
    return (base_tensor, first_operand)

@torch.fx.wrap
def optimize_repeated_broadcast(base_tensor, first_operand):
    """Optimize repeated broadcasting by computing it once"""
    # Compute the broadcasted tensor only once
    broadcasted = base_tensor.unsqueeze(1).unsqueeze(0)
    
    # Use it for both additions
    result = first_operand + broadcasted + broadcasted
    return result

def replacement_func():
    return optimize_repeated_broadcast