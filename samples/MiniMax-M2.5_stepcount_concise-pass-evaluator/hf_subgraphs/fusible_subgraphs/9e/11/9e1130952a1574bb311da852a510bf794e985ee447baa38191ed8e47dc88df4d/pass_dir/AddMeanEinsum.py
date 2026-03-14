import torch


def pattern(in_4, in_5):
    """
    Pattern: tmp_4 = in_5 + in_4; tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    """
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    return tmp_5


def replacement_args(in_4, in_5):
    return (in_4, in_5)


def replacement_func():
    def optimized_add_mean_einsum(in_4, in_5):
        """
        Using einsum for potentially better memory access patterns.
        Computes mean over spatial dimensions (2,3).
        """
        # einsum: sum over h,w for each b,c
        # This is equivalent to: (in_4 + in_5).mean(dim=(2,3))
        return torch.einsum('bchw->bc', in_4 + in_5) / (in_4.shape[2] * in_4.shape[3])
    
    return optimized_add_mean_einsum