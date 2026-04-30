from pass_dir.shared_kernels import shared_dispatch


# Pattern matching function
def pattern(in_0, in_1):
    tmp_1 = in_0 * 1000000.0
    tmp_2 = in_1 - tmp_1
    return tmp_2


# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1, "masked_sub")


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return shared_dispatch