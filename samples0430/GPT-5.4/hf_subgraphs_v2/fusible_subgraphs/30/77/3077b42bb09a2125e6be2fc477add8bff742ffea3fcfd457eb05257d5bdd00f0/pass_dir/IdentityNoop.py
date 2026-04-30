from pass_dir._shared_fused_silu_add import shared_replacement_func


def pattern(x):
    return x


def replacement_args(x):
    return (x, x)


def replacement_func():
    return shared_replacement_func()