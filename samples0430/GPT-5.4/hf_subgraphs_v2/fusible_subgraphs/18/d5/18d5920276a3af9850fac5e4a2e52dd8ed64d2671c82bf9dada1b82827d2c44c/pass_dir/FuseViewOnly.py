from pass_dir.shared_relu_flatten import replacement_func as _shared_replacement_func


def pattern(in_0):
    tmp_0 = in_0.view(in_0.shape[0], -1)
    return tmp_0


def replacement_args(in_0):
    return (in_0, "view_only")


def replacement_func():
    return _shared_replacement_func()