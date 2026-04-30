from pass_dir.shared_fused_ops import shared_runtime_dispatch, shared_replacement_func


def pattern(in_0, in_3):
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.expand_as(in_3)
    tmp_7 = tmp_6.float()
    return tmp_7


def replacement_args(in_0, in_3):
    return (in_0, None, None, in_3, "mask_expand_float")


def replacement_impl(in_0, in_3):
    return shared_runtime_dispatch(in_0, None, None, in_3, "mask_expand_float")


def replacement_func():
    return shared_replacement_func()