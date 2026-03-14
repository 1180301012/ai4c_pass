import torch

def pattern(x):
    """Pattern matching for permute(0, 2, 1, 3) -> contiguous -> view transformation"""
    t = x.permute(0, 2, 1, 3)
    t_cont = t.contiguous()
    result = t_cont.view(4, 512, 32)  # Match specific shape for graph 5
    return result

def replacement_args(x):
    """Extract arguments for the replacement function"""
    return (x,)

@torch.fx.wrap
def fused_permute_view(x):
    """Optimized version that skips redundant intermediate steps"""
    # The original does: permute -> contiguous -> view
    # We can optimize by skipping explicit contiguous call and using reshape
    # which handles memory layout more efficiently
    result = x.permute(0, 2, 1, 3).reshape(4, 512, 32)
    return result

def replacement_func():
    """Return the optimized implementation"""
    return fused_permute_view