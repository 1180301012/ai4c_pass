import torch

# Pattern matching to detect and eliminate dropout with p=0.0
def pattern(bmm_result):
    # Softmax followed by dropout with p=0.0 (no-op)
    tmp_1 = torch.nn.functional.softmax(bmm_result, dim=-1)
    tmp_2 = torch.nn.functional.dropout(tmp_1, p=0.0, training=False)
    return tmp_1, tmp_2

# Arguments needed for replacement
def replacement_args(bmm_result):
    return (bmm_result,)

# Optimized function that eliminates dropout
def optimized_dropout_elimination(bmm_result):
    # Since dropout with p=0.0 is mathematically equivalent to identity,
    # we can just return the softmax result directly
    return torch.nn.functional.softmax(bmm_result, dim=-1)

# Wrapper function for the replacement
@torch.fx.wrap
def dropout_elimination_wrapper(bmm_result):
    return optimized_dropout_elimination(bmm_result)

# Replacement function (returns function reference, not a call)
def replacement_func():
    return dropout_elimination_wrapper