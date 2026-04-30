from pass_dir.shared_kernels import fused_add_dispatch, pattern_add2

pattern = pattern_add2


def replacement_args(in_0, in_1):
    return (in_0, in_1, "add2")


def replacement_func():
    return fused_add_dispatch