from pass_dir.shared_fused_ops import shared_runtime_dispatch, shared_replacement_func


def pattern(tmp_7, tmp_4):
    tmp_8 = tmp_4 * tmp_7
    return tmp_8


def replacement_args(tmp_7, tmp_4):
    return (tmp_7, None, None, tmp_4, "mul")


def replacement_impl(tmp_7, tmp_4):
    return shared_runtime_dispatch(tmp_7, None, None, tmp_4, "mul")


def replacement_func():
    return shared_replacement_func()