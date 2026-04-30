from pass_dir.shared_kernels import fused_add_dispatch, pattern_add3_v2

pattern = pattern_add3_v2


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "add3_v2")


def replacement_func():
    return fused_add_dispatch