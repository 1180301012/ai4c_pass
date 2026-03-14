import torch

# Pattern: in_4 + in_3 followed by dropout2d with training=False
# When training=False, dropout2d is a no-op (returns input unchanged)
# So we can replace the whole pattern with just the add operation (no kernel needed)
def pattern(in_3, in_4):
    """
    Match the add + dropout2d pattern.
    dropout2d with training=False and inplace=False just returns the input unchanged.
    """
    tmp_3 = in_4 + in_3
    tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    return tmp_4


def replacement_args(in_3, in_4):
    # Return the add inputs directly
    return (in_3, in_4)


def simple_add(x, y):
    """Simple addition - since dropout is a no-op in inference, we just need the add"""
    return x + y


def replacement_func():
    return simple_add