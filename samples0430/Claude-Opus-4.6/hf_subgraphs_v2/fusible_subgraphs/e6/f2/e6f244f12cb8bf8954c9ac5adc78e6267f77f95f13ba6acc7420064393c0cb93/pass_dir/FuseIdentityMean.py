from pass_dir.shared_kernels import fused_add_dispatch, pattern_identity

pattern = pattern_identity


def replacement_args(in_0):
    return (in_0, "identity")


def replacement_func():
    return fused_add_dispatch