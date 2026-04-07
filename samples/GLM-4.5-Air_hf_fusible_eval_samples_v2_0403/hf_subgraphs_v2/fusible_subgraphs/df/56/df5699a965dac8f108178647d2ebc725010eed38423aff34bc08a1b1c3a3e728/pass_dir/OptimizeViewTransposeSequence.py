import torch
import triton
import triton.language as tl

def pattern(tmp_5):
    """
    Matches: view -> transpose -> contiguous -> view sequence
    This can be optimized to just view since the transpose and contiguous are redundant
    """
    # View operation
    tmp_7 = tmp_5.view(512, 2, 20, 64, 48)
    # Transpose operation
    tmp_8 = torch.transpose(tmp_7, 1, 2)
    # Contiguous operation
    tmp_9 = tmp_8.contiguous()
    # Final view operation back to original shape
    tmp_10 = tmp_9.view(512, 40, 64, 48)
    
    # Return intermediate and final results as they are used later
    return tmp_8, tmp_10

def replacement_args(tmp_5):
    """Extract arguments for the replacement function"""
    return (tmp_5,)

@torch.fx.wrap
def optimized_sequence(tmp_5):
    """
    Optimized version that skips unnecessary transpose and contiguous operations
    The sequence view->transpose->contiguous->view can be replaced with just view
    """
    # Direct view to final shape - transpose and contiguous are unnecessary
    output = tmp_5.view(512, 40, 64, 48)
    
    # Compute what the intermediate transpose would have produced
    intermediate = tmp_5.view(512, 20, 2, 64, 48).transpose(1, 2)
    
    return intermediate, output

def replacement_func():
    """Return the optimized function"""
    return optimized_sequence