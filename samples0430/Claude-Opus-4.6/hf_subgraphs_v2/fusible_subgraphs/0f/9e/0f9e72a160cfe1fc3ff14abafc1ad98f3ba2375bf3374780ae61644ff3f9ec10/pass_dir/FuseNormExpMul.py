from pass_dir.shared_dispatch import dispatch


def pattern(x):
    n = x.norm(p=2, dim=-1, keepdim=True)
    result = x / n
    return result


def replacement_args(x):
    return ("norm_div", x)


def replacement_func():
    return dispatch