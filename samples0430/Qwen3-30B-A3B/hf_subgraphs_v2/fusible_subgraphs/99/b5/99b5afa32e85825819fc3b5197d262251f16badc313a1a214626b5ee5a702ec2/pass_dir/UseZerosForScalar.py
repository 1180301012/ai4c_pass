import torch

def pattern(device):
    tmp_0 = torch.arange(1, device=device)
    _ = torch._functorch.vmap.lazy_load_decompositions()
    return tmp_0

def replacement_args(device):
    return (device,)

@torch.fx.wrap
def zeros_for_scalar(device):
    return torch.zeros(1, device=device)

def replacement_func():
    return zeros_for_scalar