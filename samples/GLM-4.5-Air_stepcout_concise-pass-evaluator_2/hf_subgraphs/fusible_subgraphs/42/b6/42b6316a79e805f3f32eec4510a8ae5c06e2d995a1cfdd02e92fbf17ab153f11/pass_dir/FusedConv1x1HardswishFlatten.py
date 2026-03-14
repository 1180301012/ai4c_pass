import torch


# Pattern matching function - matches Conv2d + Hardswish + Flatten
def pattern(in_0, in_1, in_2):
    """
    Pattern: Conv2D (1x1) + Hardswish + Flatten
    """
    # Conv2D with 1x1 kernel
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    # Hardswish activation
    tmp_3 = torch.nn.functional.hardswish(tmp_2, True)
    # Flatten from dim 1
    tmp_4 = tmp_3.flatten(1, -1)
    return tmp_4


# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    """
    Returns a pass-through function that returns None.
    This allows the original computation to run without modification.
    """
    def pass_through(in_0, in_1, in_2):
        # Return None to indicate no replacement - original graph runs
        return None
    
    return pass_through