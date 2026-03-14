import torch

@torch.fx.wrap
def optimize_expand_concat(cls_token, patches):
    """
    Optimize expand + concat operations by doing them in one step
    Instead of expand then concat, we can directly concatenate
    """
    # cls_token: [1, 1, 768] -> we want [1, 1, 768] concatenated with patches [1, 196, 768]
    # Result should be: [1, 197, 768]
    
    # Use torch.cat directly - this is more efficient than separate expand + concat
    return torch.cat((cls_token, patches), dim=1)

# Pattern matching function
def pattern(cls_token, patches):
    """
    Match: expand + concat operations
    """
    tmp_8 = cls_token.expand(1, -1, -1)
    tmp_9 = torch.cat((tmp_8, patches), dim=1)
    return tmp_9

# Argument extraction function
def replacement_args(cls_token, patches):
    return (cls_token, patches)

# Replacement function
def replacement_func():
    return optimize_expand_concat