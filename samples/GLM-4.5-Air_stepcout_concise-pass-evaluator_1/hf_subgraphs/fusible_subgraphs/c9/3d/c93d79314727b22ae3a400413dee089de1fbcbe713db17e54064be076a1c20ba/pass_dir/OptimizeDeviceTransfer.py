import torch

# Pattern matching function - match both scalar device transfers
def pattern(in_0, in_1):
    """Match sequential device transfers for scalars"""
    # This pattern matches the structure:
    # tmp_3 = tmp_1.to(device(type='cuda'))
    # tmp_4 = tmp_0.to(device(type='cuda'))
    # where tmp_0 = in_0, tmp_1 = in_1
    tmp_3 = in_1.to(device(type='cuda'))
    tmp_4 = in_0.to(device(type='cuda'))
    return tmp_3, tmp_4

# Replacement arguments function  
def replacement_args(in_0, in_1):
    """Extract the scalar tensors for device transfer optimization"""
    return (in_0, in_1)

# Replacement function - optimize device transfers with proper CUDA device setting
def replacement_func():
    """Optimize scalar device transfers using direct CUDA tensor creation"""
    def optimized_device_transfer(in_0, in_1):
        # For scalar transfers, create new tensors directly on CUDA device
        # This avoids potential overhead from .to() method calls while achieving the same result
        tmp_3 = torch.tensor(in_1, dtype=in_1.dtype, device='cuda')
        tmp_4 = torch.tensor(in_0, dtype=in_0.dtype, device='cuda')
        return tmp_4, tmp_3
    
    return optimized_device_transfer