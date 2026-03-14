import torch


def pattern(in_4, in_5):
    """
    Pattern: tmp_4 = in_5 + in_4; tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    
    Mathematical transformation:
    (in_5 + in_4).mean(dim=(2,3)) = in_5.mean(dim=(2,3)) + in_4.mean(dim=(2,3))
    """
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    return tmp_5


def replacement_args(in_4, in_5):
    return (in_4, in_5)


def replacement_func():
    def optimized_add_mean(in_4, in_5):
        return in_4.mean(dim=(2, 3), keepdim=False) + in_5.mean(dim=(2, 3), keepdim=False)
    
    return optimized_add_mean