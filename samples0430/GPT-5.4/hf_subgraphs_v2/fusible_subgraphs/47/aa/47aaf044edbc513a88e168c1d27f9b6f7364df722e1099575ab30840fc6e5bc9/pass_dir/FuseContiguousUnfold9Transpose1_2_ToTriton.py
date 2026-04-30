from pass_dir.convbert_shared import dispatch_convbert_replacement


def pattern(in_0):
    tmp_0 = in_0.transpose(1, 2)
    return tmp_0


def replacement_args(in_0):
    return (in_0, "transpose")


def replacement_func():
    return dispatch_convbert_replacement