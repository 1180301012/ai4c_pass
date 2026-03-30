import torch

def pattern(tensor1, tensor2, tensor3, target_shape):
    """Pattern matching for concatenation followed by reshape"""
    concat_result = torch.cat([tensor1, tensor2, tensor3], dim=1)
    reshape_result = concat_result.reshape(target_shape)
    return reshape_result

def replacement_args(tensor1, tensor2, tensor3, target_shape):
    """Extract arguments for the optimized cat+reshape operation"""
    return (tensor1, tensor2, tensor3, target_shape)

@torch.fx.wrap
def optimized_cat_reshape(tensor1, tensor2, tensor3, target_shape):
    """Wrapper function for optimized concatenation + reshape"""
    # Use torch.cat directly as it's already highly optimized
    # The reshape operation will be handled by PyTorch's optimized backend
    concat_result = torch.cat([tensor1, tensor2, tensor3], dim=1)
    reshape_result = concat_result.reshape(target_shape)
    return reshape_result

def replacement_func():
    """Return the optimized cat+reshape function"""
    return optimized_cat_reshape