import torch


def pattern(x):
    """
    Match pattern: dropout(dropout(x, 0.0, ...), 0.0, ...)
    When probability is 0.0, dropout is a no-op and returns input unchanged.
    Two consecutive no-op dropouts can be eliminated entirely with one replacement.
    """
    tmp1 = torch.nn.functional.dropout(x, 0.0, False, False)
    tmp2 = torch.nn.functional.dropout(tmp1, 0.0, False, False)
    return tmp2


def replacement_args(x):
    return (x,)


@torch.fx.wrap
def dropout0_identity(x):
    """
    Since dropout(0.0) is a no-op (identity), we simply return the input directly.
    """
    return x


def replacement_func():
    return dropout0_identity