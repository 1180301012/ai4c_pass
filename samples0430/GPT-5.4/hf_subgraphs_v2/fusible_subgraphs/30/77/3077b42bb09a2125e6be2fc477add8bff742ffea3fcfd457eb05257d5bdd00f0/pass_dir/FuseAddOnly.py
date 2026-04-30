from pass_dir._shared_fused_silu_add import shared_replacement_func


def pattern(x, y):
    tmp = x + y
    return tmp


def replacement_args(x, y):
    return (x, y, "add_only")


def replacement_func():
    return shared_replacement_func()