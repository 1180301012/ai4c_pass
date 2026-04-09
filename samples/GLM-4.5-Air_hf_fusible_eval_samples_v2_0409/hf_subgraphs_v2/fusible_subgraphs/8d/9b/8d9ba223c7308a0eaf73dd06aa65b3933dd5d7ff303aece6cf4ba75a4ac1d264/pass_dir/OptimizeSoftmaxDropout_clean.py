import torch

@torch.fx.wrap
def optimized_softmax(input_tensor, dim=-1):
    # Since dropout rate is 0.0, we just return softmax directly
    # This avoids the unnecessary overhead of calling dropout with rate=0
    return torch.softmax(input_tensor, dim=dim)

def pattern(viewed_tensor, softmax_out, dropout_out):
    # Pattern matching: view -> softmax -> dropout(0.0)
    # tmp_20 = tmp_19.view(-1, 12, 64, 64)
    # tmp_21 = torch.nn.functional.softmax(tmp_20, dim = -1)
    # tmp_22 = torch.nn.functional.dropout(tmp_21, 0.0, False, False)
    
    # Since dropout rate is 0.0, it's essentially a no-op
    tmp_21 = torch.nn.functional.softmax(viewed_tensor, dim=-1)
    tmp_22 = torch.nn.functional.dropout(tmp_21, 0.0, False, False)
    return tmp_22

def replacement_args(viewed_tensor):
    return (viewed_tensor,)

def replacement_func():
    return optimized_softmax