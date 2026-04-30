from pass_dir.shared_kernels import shared_dispatch


# Pattern matching function
def pattern(x):
    tmp_1 = x.squeeze(-1)
    tmp_2 = tmp_1.contiguous()
    return tmp_2


# Argument extraction function
def replacement_args(x):
    return (x, "squeeze_contiguous")


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return shared_dispatch